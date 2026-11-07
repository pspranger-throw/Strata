#!/usr/bin/env python3
"""verify_exl3_pack.py — G0 gate for a pack produced by tools/exl3/pack_exl3.py.

Independence rule (nvfp4 pattern, 31-B §2): the verifier re-derives every pack byte position from
the SOURCE pack's safetensors headers and the layout rules recorded in the pack's own text tables
(native_experts.txt / exl3_dense.txt / index.txt), WITHOUT importing pack_exl3.py, and compares
pack vs source region by region: experts.bin (all 24,576 blobs, every byte), dense.bin (every
region), the ngram byte-for-byte copy, the metadata carrier, the manifest completion marker and
the 41-census count assertions.  Finally it decodes a sample of real tensors bit-exactly TWICE -
from the source tensors and from the pack regions - with the reviewed numpy reference (the sociolog
clone's tools/exl3 package, used read-only as an external checker; the fork C++ reference already
passed on these bytes in the A3' arm, doc 42).  Any mismatch exits nonzero.

Usage: verify_exl3_pack.py --src <hf exl3 dir> --pack <strata pack dir> [--ref-tools <fork tools>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import struct

import numpy as np

ALIGN = 64
ROLES = ("gate_proj", "up_proj", "down_proj")
NP_DTYPE = {"I16": "<i2", "F16": "<f2", "I32": "<u4"}


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def read_st(f: pathlib.Path) -> dict:
    """safetensors header -> {key: (dtype, shape, off0, off1)}."""
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
            for k, v in h.items()}


class Raw:
    """Bounded-RSS raw reader over the source shards (file handles cached, no tensor store)."""

    def __init__(self, src: pathlib.Path):
        self.src = src
        self.tmap, self.key_file, self.base = {}, {}, {}
        for p in sorted(src.glob("*.safetensors")):
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.base[p.name] = 8 + n
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k, v in h.items():
                self.tmap[k] = (v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
                self.key_file[k] = p.name
        self._fh = {}

    def __getitem__(self, key: str) -> bytes:
        fname = self.key_file[key]
        fh = self._fh.get(fname)
        if fh is None:
            fh = self._fh[fname] = open(self.src / fname, "rb")
        _dt, _shp, o0, o1 = self.tmap[key]
        fh.seek(self.base[fname] + o0)
        b = fh.read(o1 - o0)
        assert len(b) == o1 - o0
        return b

    def np(self, key: str) -> np.ndarray:
        dt, shp, _o0, _o1 = self.tmap[key]
        return np.frombuffer(self[key], dtype=NP_DTYPE[dt]).reshape(shp)

    def size(self, key: str) -> int:
        return self.tmap[key][3] - self.tmap[key][2]


def sha256_file(p: pathlib.Path, blk: int = 1 << 20) -> str:
    hh = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(blk):
            hh.update(b)
    return hh.hexdigest()


def blob_part_offsets(K: int) -> dict:
    """Within an expert blob, parts are written contiguously (trellis then suh then svh per role;
    all sizes are already 64-multiples), then the 16 B mul1 tail, then zero pad to the stride."""
    off, out = 0, {}
    for r in ROLES:
        for part, size in (("trellis", 204_800 * K), ("suh", None), ("svh", None)):
            out[(r, part)] = off
            off += size if size is not None else SUH_SIZE[(r, part)]
    out[("tail", "mul1")] = off
    return out


SUH_SIZE = {("gate_proj", "suh"): 5120, ("gate_proj", "svh"): 1280,
            ("up_proj", "suh"): 5120, ("up_proj", "svh"): 1280,
            ("down_proj", "suh"): 1280, ("down_proj", "svh"): 5120}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path)
    ap.add_argument("--pack", required=True, type=pathlib.Path)
    ap.add_argument("--ref-tools", type=pathlib.Path, default=None,
                    help="directory containing the fork's exl3 numpy package; decode parity skipped if absent")
    ap.add_argument("--sample", type=int, default=4, help="expert decodes per K class")
    a = ap.parse_args()
    src, pack = a.src, a.pack
    raw = Raw(src)
    t = raw.tmap
    fails = []

    def check(cond: bool, msg: str):
        print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            fails.append(msg)

    # ---- census (41 authority), independently ----
    ngram_file = next(p.name for p in src.glob("ngram*.safetensors"))
    ngram_keys = set(read_st(src / ngram_file))
    counts = dict(routed=0, dense=0, visual=0, mtp=0, ngram=0)
    for k in t:
        if k in ngram_keys:
            counts["ngram"] += 1
        elif k.startswith("model.visual."):
            counts["visual"] += 1
        elif k.startswith("mtp."):
            counts["mtp"] += 1
        elif ".mlp.experts." in k:
            counts["routed"] += 1
        else:
            counts["dense"] += 1
    print("census:", counts)
    check(counts == dict(routed=294_912, dense=2_006, visual=987, mtp=6_203, ngram=132), "41 count census")
    mp = pack / "manifest.json"
    if not mp.exists():
        print("FAIL: no manifest.json - pack incomplete")
        return 1
    man = json.load(open(mp))
    check(man["counts"] == counts, "manifest counts == independent census")

    # ---- experts.bin: every byte re-derived from source headers + v6 table ----
    ne = (pack / "native_experts.txt").read_text().splitlines()
    check(any(l.startswith("# strata native experts v6") for l in ne if l.startswith("#")),
          "native_experts v6 header")
    lines = [l.split() for l in ne if l and not l.startswith("#")]
    check(len(lines) == 48, f"48 layer lines ({len(lines)})")
    geo = {int(l[0]): (int(l[3]), int(l[4]), int(l[5])) for l in lines}
    check(all(int(l[1]) == 200 and int(l[2]) == 200 for l in lines), "format ids 200 on every line")
    check(sum(geo[l][1] * 512 for l in geo) == (pack / "experts.bin").stat().st_size,
          "experts.bin size == sum(blob_bytes x n_expert)")
    n_bad = 0
    with open(pack / "experts.bin", "rb") as pf:
        for l in range(48):
            off, stride, K = geo[l]
            assert stride == align_up(614_400 * K + 19_216)
            pofs = blob_part_offsets(K)
            for e in range(512):
                pre = f"model.language_model.layers.{l}.mlp.experts.{e}."
                want = bytearray()
                for r in ROLES:
                    for part in ("trellis", "suh", "svh"):
                        want += raw[pre + r + "." + part]
                muls = [struct.unpack("<i", raw[pre + r + ".mul1"])[0] for r in ROLES]
                want += struct.pack("<4i", *muls, 0)
                got = pf.read(stride)
                if got != bytes(want) + b"\x00" * (stride - len(want)):
                    n_bad += 1
                    if n_bad <= 4:
                        print(f"  FAIL experts.bin L{l} E{e}", flush=True)
            if l % 12 == 0 or l == 47:
                print(f"  experts verified through layer {l} ({n_bad} bad)", flush=True)
    check(n_bad == 0, f"experts.bin every byte ({n_bad} blobs bad)")

    # ---- dense.bin ----
    dense_src = {k for k in t if k not in ngram_keys and not k.startswith("model.visual.")
                 and not k.startswith("mtp.") and ".mlp.experts." not in k}
    groups: dict[str, dict] = {}
    for k in dense_src:
        for suf in (".trellis", ".suh", ".svh", ".mul1"):
            if k.endswith(suf):
                groups.setdefault(k[: -len(suf)], {})[suf[1:]] = k
                break
        else:
            groups.setdefault(k, {})
    # ---- 46 canonical-vocab re-derivation (independent copy of the layout rules) ----
    BF16, F32, F16, EXL3 = "BF16", "F32", "F16", "exl3"
    L46 = {
        "attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", BF16),
        "attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", BF16),
        "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", BF16),
        "attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", F32),
        "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", BF16),
        "mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", BF16),
        "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", BF16),
        "mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", F32),
        "mlp.gate.weight": ("ffn_gate_inp.weight", BF16),
        "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", BF16),
        "mlp.shared_expert.gate_proj": ("ffn_gate_shexp.weight", EXL3),
        "mlp.shared_expert.up_proj": ("ffn_up_shexp.weight", EXL3),
        "mlp.shared_expert.down_proj": ("ffn_down_shexp.weight", EXL3),
        "linear_attn.in_proj_qkv": ("attn_qkv.weight", EXL3),
        "linear_attn.in_proj_z": ("attn_gate.weight", EXL3),
        "linear_attn.out_proj": ("ssm_out.weight", EXL3),
        "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", BF16),
        "linear_attn.in_proj_b.weight": ("ssm_beta.weight", BF16),
        "linear_attn.A_log": ("ssm_a", F32), "linear_attn.dt_bias": ("ssm_dt.bias", F32),
        "linear_attn.norm.weight": ("ssm_norm.weight", F32),
        "linear_attn.conv1d.weight": ("ssm_conv1d.weight", F32),
        "self_attn.q_proj": ("attn_q.weight", EXL3), "self_attn.k_proj": ("attn_k.weight", EXL3),
        "self_attn.v_proj": ("attn_v.weight", EXL3), "self_attn.o_proj": ("attn_output.weight", EXL3),
        "self_attn.q_norm.weight": ("attn_q_norm.weight", F32),
        "self_attn.k_norm.weight": ("attn_k_norm.weight", F32),
        "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", F32),
        "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", F32),
        "self_attn.indexer.index_qk_proj": ("__split__", EXL3),
        "ple.key_proj.weight": ("ple_key.weight", BF16),
        "ple.value_proj.weight": ("ple_value.weight", BF16),
        "ple.norm_key.weight": ("ple_norm_key.weight", F32),
        "ple.norm_query.weight": ("ple_norm_query.weight", F32),
        "ple.norm_conv.weight": ("ple_norm_conv.weight", F32),
        "ple.conv1d.weight": ("ple_conv1d.weight", F16),
    }

    def canon46(key: str) -> tuple[str, str]:
        if key == "model.language_model.embed_tokens.weight":
            return "token_embd.weight", "shape"
        if key == "lm_head":
            return "output.weight", EXL3
        if key == "model.language_model.hyper_connection_mixer.hc_norm.weight":
            return "output_hc_norm.weight", F32
        if key == "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight":
            return "output_hc_down.weight", BF16
        if key == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight":
            return "output_hc_up.weight", BF16
        m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", key)
        if m and m.group(2) in L46:
            hit = L46[m.group(2)]
            if hit[0] == "__split__":
                return f"blk.{m.group(1)}.indexer", "exl3-split"
            return f"blk.{m.group(1)}.{hit[0]}", hit[1]
        raise KeyError(f"46 verifier: no canonical name for {key}")

    def conv46(b: bytes, src_dt: str, form: str) -> bytes:
        if form == src_dt:
            return b
        if form == F32 and src_dt == "BF16":
            return (np.frombuffer(b, "<u2").astype("<u4") << 16).view("<f4").tobytes()
        if form == F32 and src_dt == "F16":
            return np.frombuffer(b, "<f2").astype("<f4").tobytes()
        if form == BF16 and src_dt == "F16":
            u = np.frombuffer(b, "<f2").astype("<f4").view("<u4")
            return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
        if form in (F16,) and src_dt in ("BF16", "F32"):
            src = (np.frombuffer(b, "<u2").astype("<u4") << 16).view("<f4") if src_dt == "BF16" \
                else np.frombuffer(b, "<f4")
            return src.astype("<f2").tobytes()
        raise AssertionError(f"46 verifier: {src_dt} -> {form}")

    dlines = [l.split() for l in (pack / "exl3_dense.txt").read_text().splitlines()
              if l and not l.startswith("#")]
    dsider = {f_[0]: f_ for f_ in dlines}
    check(len(dlines) == sum(1 for p_ in groups.values() if p_) - 12 + 24,
          f"exl3_dense rows == source groups - 12 fused + 24 split ({len(dlines)})")
    n_bad = 0
    with open(pack / "dense.bin", "rb") as df:
        def at(off_: int, n_: int) -> bytes:
            df.seek(int(off_))
            return df.read(n_)
        for base, parts in groups.items():
            cname, form = canon46(base)
            if not parts:
                continue
            if form == "exl3-split":
                for suffix, t0, t1, o0, o1 in (("q_proj.weight", 0, 32, 0, 512),
                                               ("k_proj.weight", 32, 40, 512, 640)):
                    f2 = dsider.get(f"{cname}.{suffix}")
                    if f2 is None:
                        n_bad += 1
                        continue
                    ki = int(f2[1]) // 16
                    tr = np.frombuffer(raw[parts["trellis"]], "<u2").reshape(ki, 40, -1)
                    if at(f2[5], tr[:, t0:t1, :].nbytes) != tr[:, t0:t1, :].tobytes():
                        n_bad += 1
                    if at(f2[6], len(raw[parts["suh"]])) != raw[parts["suh"]]:
                        n_bad += 1
                    if at(f2[7], 2 * (o1 - o0)) != raw[parts["svh"]][2 * o0:2 * o1]:
                        n_bad += 1
                    if int(f2[8]) != struct.unpack("<i", raw[parts["mul1"]])[0]:
                        n_bad += 1
                continue
            f2 = dsider.get(cname)
            if f2 is None:
                n_bad += 1
                continue
            for part, col in (("trellis", 5), ("suh", 6), ("svh", 7)):
                if at(f2[col], len(raw[parts[part]])) != raw[parts[part]]:
                    n_bad += 1
            if int(f2[8]) != struct.unpack("<i", raw[parts["mul1"]])[0]:
                n_bad += 1
    check(n_bad == 0, f"exl3_dense every region incl split ({n_bad} bad fields)")
    ilines = [l.split() for l in (pack / "index.txt").read_text().splitlines()
              if l and not l.startswith("#")]
    isider = {f_[0]: f_ for f_ in ilines}
    n_bad = 0
    with open(pack / "dense.bin", "rb") as df:
        for base, parts in groups.items():
            if parts:
                continue
            cname, form = canon46(base)
            f2 = isider.get(cname)
            if f2 is None:
                n_bad += 1
                continue
            if form == "shape":
                if f2[9] != "8" or f2[11] != "32":
                    n_bad += 1
                    print(f"  shape row wrong: {cname} fields {f2[9]}/{f2[11]}")
                continue
            want = conv46(raw[base], raw.tmap[base][0], form)
            df.seek(int(f2[3]))
            got = df.read(len(want))
            if got != want:
                n_bad += 1
                print(f"  index byte mismatch: {base} -> {cname} (src dtype {raw.tmap[base][0]}, "
                      f"form {form}, {len(want)} B)")
    check(n_bad == 0, f"index rows every byte incl form conversion ({n_bad} bad)")
    check(len(ilines) == sum(1 for p_ in groups.values() if not p_),
          f"index rows == source plain tensors ({len(ilines)})")

    # ---- ngram + metadata + manifest shas ----
    check(sha256_file(src / ngram_file) == sha256_file(pack / ngram_file), "ngram sha256")
    mg = (pack / "metadata.gguf").read_bytes()
    check(mg[:4] == b"GGUF" and struct.unpack("<Q", mg[8:16])[0] == 0, "metadata.gguf magic + n_tensors=0")
    for f_, rec in man["files"].items():
        p_ = pack / f_
        check(p_.stat().st_size == rec["bytes"] and sha256_file(p_) == rec["sha256"], f"manifest sha {f_}")

    # ---- decode parity: source vs pack, through the reviewed numpy reference ----
    if a.ref_tools is not None:
        import sys
        sys.path.insert(0, str(a.ref_tools))
        import exl3
        from exl3.codebook import CB_MUL1

        class Region:
            def __init__(self, fh, off: int, key: str):
                dt, shp, _o0, _o1 = t[key]
                fh.seek(off)
                self.a = np.frombuffer(fh.read(_o1 - _o0), dtype=NP_DTYPE[dt]).reshape(shp)

        dec = {2: 0, 3: 0}
        with open(pack / "experts.bin", "rb") as pf:
            for l in range(48):
                off, stride, K = geo[l]
                if dec[K] >= a.sample:
                    continue
                dec[K] += 1
                e = 0
                base = off + e * stride
                pre = f"model.language_model.layers.{l}.mlp.experts.{e}."
                w_src = exl3.reconstruct_weight(raw.np(pre + "gate_proj.trellis"),
                                                raw.np(pre + "gate_proj.suh"),
                                                raw.np(pre + "gate_proj.svh"), CB_MUL1)
                pofs = blob_part_offsets(K)
                tr = Region(pf, base + pofs[("gate_proj", "trellis")], pre + "gate_proj.trellis").a
                su = Region(pf, base + pofs[("gate_proj", "suh")], pre + "gate_proj.suh").a
                sv = Region(pf, base + pofs[("gate_proj", "svh")], pre + "gate_proj.svh").a
                w_pk = exl3.reconstruct_weight(tr, su, sv, CB_MUL1)
                check(np.array_equal(w_src, w_pk),
                      f"decode parity L{l} E{e} gate K{K} bit-exact {w_src.shape}")
        with open(pack / "dense.bin", "rb") as df:
            # 46: dense sidecar rows carry CANONICAL names now — walk source groups and map them
            # through canon46 (the verifier's own rule copy) instead of matching v1 HF names.
            dsider2 = {f_[0]: f_ for f_ in dlines}
            for base_, parts_ in groups.items():
                if not parts_:                              # plain groups: no exl3 parts (base names
                    continue                                 # are NOT in the header map — only part keys are)
                cname, form_ = canon46(base_)
                if form_ != "exl3" or not (cname == "output.weight"
                                           or cname.endswith(".ffn_gate_shexp.weight")):
                    continue
                f_ = dsider2[cname]
                w_src = exl3.reconstruct_weight(raw.np(parts_["trellis"]), raw.np(parts_["suh"]),
                                                raw.np(parts_["svh"]), CB_MUL1)
                tr = Region(df, int(f_[5]), parts_["trellis"]).a
                su = Region(df, int(f_[6]), parts_["suh"]).a
                sv = Region(df, int(f_[7]), parts_["svh"]).a
                w_pk = exl3.reconstruct_weight(tr, su, sv, CB_MUL1)
                check(np.array_equal(w_src, w_pk), f"decode parity dense {cname} K{f_[3]} bit-exact {w_src.shape}")

    print("FAILURES:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
