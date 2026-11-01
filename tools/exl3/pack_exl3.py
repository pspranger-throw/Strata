#!/usr/bin/env python3
"""pack_exl3.py — convert an HF exl3 safetensors pack into a strata native pack.

Phase-1 tool of the exl3-native line (design: issues strata-exl3 90 spec §2, blob layout 31-B §1,
census authority 41 §1).  The conversion is a LOSSLESS PERMUTATION: every stored word of every
trellis/suh/svh/mul1 tensor is copied verbatim; no requantization, no folding, no value conversion
happens here.  tools/exl3/verify_exl3_pack.py re-derives the layout independently from the source
headers, checks every byte against the pack, and runs bit-exact decode parity against the numpy
reference on a sample.

Output pack (one directory):
  experts.bin          48 layers x 512 blobs, role-major per the dsv41k exl3 ExpertLayout contract:
                     [gate trellis|gate suh|gate svh|up ...|down ...] + 16 B I32 tail
                     {mul1_gate, mul1_up, mul1_down, 0}; each blob padded to 64 B.
  native_experts.txt v6: "layer gu_type d_type offset blob_bytes K" (200 = kTypeEXL3,
                     include/strata/dsv41/exl3.hpp:30); offsets inside experts.bin.
  dense.bin            every non-expert LM tensor, raw, 64 B aligned regions; the exl3 linears'
                     trellis/suh/svh/mul1 ride the same file at sidecar-recorded offsets.
  exl3_dense.txt       sidecar for the dense linears: name ne0 ne1 K cb trellis_off suh_off svh_off
                     mul1_val.
  index.txt            v3 schema (weights.cpp), rows for the F16/BF16 tensors only - the kinds the
                     generic loader serves; kind 4 = BF16, 5 = F16 (tools/iq_pack.py KIND).
  ngram_embedding.safetensors  copied byte-for-byte (runtime-mmap PLE; 31-B hybrid rule).
  metadata.gguf        GGUF metadata-only carrier: the KV section of a same-architecture GGUF,
                     byte-identical, header rewritten with n_tensors=0 (the --native seam).
  manifest.json        written LAST as the completion marker (iq_pack.py discipline): counts,
                     sha256s, per-layer K, blob strides, provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import struct
import sys

ALIGN = 64
N_EXPERT = 512
ROLES = ("gate_proj", "up_proj", "down_proj")
TRELLIS_TAIL = 16  # {mul1_gate, mul1_up, mul1_down, 0} I32; mirrors iq_pack NVFP4_TAIL
K_TYPE_EXL3 = 200  # reserved format id, include/strata/dsv41/exl3.hpp:30
KIND = {"BF16": 4, "F16": 5, "F32": 2}  # index.txt kinds, tools/iq_pack.py:90

# C6.0 (issues 46): the engine's closed vocabulary (src/core/layout.cpp:50-107, ground truth the
# iq3_s pack) speaks strata names and forms, not the HF checkpoint's.  canon() maps every dense
# source key to its canonical name + engine FORM (iq_pack.py FORM table); anything unmapped is a
# hard error - an unmapped tensor is a boot-time "missing blk.N.*" waiting to happen.
import re
import numpy as np

BF16, F32, F16 = "BF16", "F32", "F16"
EXL3 = "exl3"

_HC_SUFFIX = {"input_mix_weight_down.weight": ("hc_attn_down.weight", BF16),
              "input_mix_weight_up.weight": ("hc_attn_up.weight", BF16),
              "block_inject_weight.weight": ("hc_attn_inject.weight", BF16),
              "hc_norm.weight": ("hc_attn_norm.weight", F32)}
_LAYER_MAP = {
    "attn_hyper_connection." + k: v for k, v in _HC_SUFFIX.items()}
_LAYER_MAP.update({"mlp_hyper_connection." + k: (v[0].replace("hc_attn_", "hc_ffn_"), v[1])
                   for k, v in _HC_SUFFIX.items()})
_LAYER_MAP.update({
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
    "linear_attn.A_log": ("ssm_a", F32),
    "linear_attn.dt_bias": ("ssm_dt.bias", F32),
    "linear_attn.norm.weight": ("ssm_norm.weight", F32),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", F32),
    "self_attn.q_proj": ("attn_q.weight", EXL3),
    "self_attn.k_proj": ("attn_k.weight", EXL3),
    "self_attn.v_proj": ("attn_v.weight", EXL3),
    "self_attn.o_proj": ("attn_output.weight", EXL3),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", F32),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", F32),
    "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", F32),
    "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", F32),
    "self_attn.indexer.index_qk_proj": ("INDEXER_SPLIT", EXL3),  # split at write time (46)
    "ple.key_proj.weight": ("ple_key.weight", BF16),
    "ple.value_proj.weight": ("ple_value.weight", BF16),
    "ple.norm_key.weight": ("ple_norm_key.weight", F32),
    "ple.norm_query.weight": ("ple_norm_query.weight", F32),
    "ple.norm_conv.weight": ("ple_norm_conv.weight", F32),
    "ple.conv1d.weight": ("ple_conv1d.weight", F16),
})


def canon(key: str) -> tuple[str, str]:
    if key == "model.language_model.embed_tokens.weight":
        return "token_embd.weight", "shape"  # carried by embd.gguf (tools/embd_bf16_pack.py)
    if key == "lm_head":
        return "output.weight", EXL3
    if key == "model.language_model.hyper_connection_mixer.hc_norm.weight":
        return "output_hc_norm.weight", F32
    if key == "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight":
        return "output_hc_down.weight", BF16
    if key == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight":
        return "output_hc_up.weight", BF16
    m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", key)
    if m:
        layer, rest = m.group(1), m.group(2)
        hit = _LAYER_MAP.get(rest)
        if hit and hit[0] == "INDEXER_SPLIT":
            return f"blk.{layer}.indexer", "exl3-split"   # q_proj/k_proj written at split time
        if hit:
            return f"blk.{layer}.{hit[0]}", hit[1]
    raise KeyError(f"46: no canonical name/form for {key}")


def convert_form(b: bytes, src_dtype: str, form: str) -> tuple[bytes, str]:
    """Cast a plain tensor's bytes to its engine FORM (46 §4.3); identity when it already matches."""
    if form == src_dtype:
        return b, form
    if form == F32 and src_dtype == "BF16":
        return ((np.frombuffer(b, "<u2").astype("<u4") << 16).view("<f4").tobytes(), F32)
    if form == F32 and src_dtype == "F16":
        return (np.frombuffer(b, "<f2").astype("<f4").tobytes(), F32)
    if form == BF16 and src_dtype == "F16":
        u = np.frombuffer(b, "<f2").astype("<f4").view("<u4")
        return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes(), BF16)
    if form == BF16 and src_dtype == "BF16":
        return b, BF16
    if form == F16 and src_dtype in ("BF16", "F32"):
        src = ((np.frombuffer(b, "<u2").astype("<u4") << 16).view("<f4") if src_dtype == "BF16"
               else np.frombuffer(b, "<f4"))
        return (src.astype("<f2").tobytes(), F16)
    raise AssertionError(f"form conversion {src_dtype} -> {form} not implemented")
# strata's GGUF dialect (include/strata/artifact/gguf_reader.hpp:322): MetaType extends the stock
# GGUF enum with U64=10, I64=11, F64=12 (stock GGUF has 10=USTRING; strata files use the 10-12 ints,
# e.g. qwen4exp.ple.head_offsets is a U64 array).  8=STRING, 9=ARRAY.
GGUF_ESZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8}


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def read_st(f: pathlib.Path) -> dict:
    """safetensors header -> {tensors: {key: {dtype, shape, off0, off1}}, meta, base}."""
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    meta = h.pop("__metadata__", None)
    return {"tensors": {k: {"dtype": v["dtype"], "shape": v["shape"],
                            "off0": v["data_offsets"][0], "off1": v["data_offsets"][1]}
                        for k, v in h.items()},
            "meta": meta, "base": 8 + n}


class Src:
    """Source pack reader: every tensor's raw bytes, straight from the shard headers."""

    def __init__(self, d: pathlib.Path):
        self.d = d
        self.headers = {p.name: read_st(p) for p in sorted(d.glob("*.safetensors"))}
        self.loc = {}  # key -> (file, off0, off1, dtype, shape)
        for fname, hd in self.headers.items():
            for k, t in hd["tensors"].items():
                self.loc[k] = (fname, t["off0"], t["off1"], t["dtype"], tuple(t["shape"]))
        self._fh = {}

    def raw(self, key: str) -> bytes:
        fname, off0, off1, _, _ = self.loc[key]
        fh = self._fh.get(fname)
        if fh is None:
            fh = self._fh[fname] = open(self.d / fname, "rb")
        fh.seek(self.headers[fname]["base"] + off0)
        b = fh.read(off1 - off0)
        assert len(b) == off1 - off0
        return b

    def nbytes(self, key: str) -> int:
        return self.loc[key][2] - self.loc[key][1]

    def close(self):
        for fh in self._fh.values():
            fh.close()


def sha256_file(p: pathlib.Path, blk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(blk):
            h.update(b)
    return h.hexdigest()


def gguf_meta_copy(src_gguf: pathlib.Path, dst: pathlib.Path) -> tuple[int, int]:
    """Copy GGUF magic/version/KV records; rewrite n_tensors to 0 and DROP split.* keys (46:
    a lone carrier must not claim to be shard 1 of 2 - validate_split throws at gguf_reader:603).
    Returns (n_tensors, n_kv_kept)."""

    def read_str(fh):
        (tl,) = struct.unpack("<Q", fh.read(8))
        return fh.read(tl)

    def walk_records(fh, n_kv: int):
        """yield (key, whole record bytes incl key); value types per strata's enum (GGUF_ESZ)."""
        for _ in range(n_kv):
            start = fh.tell()
            key = read_str(fh).decode("utf-8")
            (t,) = struct.unpack("<I", fh.read(4))
            if t == 8:
                read_str(fh)
            elif t == 9:
                (at,) = struct.unpack("<I", fh.read(4))
                (an,) = struct.unpack("<Q", fh.read(8))
                if at == 8:
                    for _ in range(an):
                        read_str(fh)
                else:
                    fh.read(an * GGUF_ESZ[at])
            else:
                fh.read(GGUF_ESZ[t])
            yield key, (start, fh.tell())

    with open(src_gguf, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF")
        ver = struct.unpack("<I", fh.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
        kv_start = fh.tell()
        kept = []
        for key, (rs, re_) in walk_records(fh, n_kv):
            if key in ("split.count", "split.no", "split.tensors.count"):
                continue
            fh.seek(rs)
            kept.append(fh.read(re_ - rs))
    with open(dst, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", ver) + struct.pack("<QQ", 0, len(kept)))
        for rec in kept:
            fh.write(rec)
    return n_tensors, len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path, help="HF exl3 pack directory")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="output strata pack directory")
    ap.add_argument("--gguf-meta", required=True, type=pathlib.Path,
                    help="same-architecture GGUF whose KV section becomes metadata.gguf")
    a = ap.parse_args()
    src, out = a.src, a.out
    out.mkdir(parents=True, exist_ok=True)
    s = Src(src)

    # ---- classify by source file + key (41 census authority; hard count assertions) ----
    ngram_file = next(name for name in s.headers if name.startswith("ngram"))
    routed, densekeys, visual, mtp, ngram = [], [], [], [], []
    for k, (fname, *_r) in s.loc.items():
        if fname == ngram_file:
            ngram.append(k)
        elif k.startswith("model.visual."):
            visual.append(k)
        elif k.startswith("mtp."):
            mtp.append(k)
        elif ".mlp.experts." in k:
            routed.append(k)
        else:
            densekeys.append(k)
    layers = sorted({int(k.split(".")[3]) for k in routed})
    experts = sorted({int(k.split(".")[6]) for k in routed})
    cnt = dict(routed=len(routed), dense=len(densekeys), visual=len(visual), mtp=len(mtp), ngram=len(ngram))
    print(f"classified {cnt} layers={len(layers)} experts={len(experts)}")
    for got, want, name in ((len(routed), 294_912, "routed"), (len(densekeys), 2_006, "dense"),
                            (len(visual), 987, "visual"), (len(mtp), 6_203, "mtp"), (len(ngram), 132, "ngram"),
                            (len(layers), 48, "layers"), (len(experts), N_EXPERT, "experts")):
        assert got == want, f"{name}: {got} != {want}"

    def K_of(key: str) -> int:
        return s.loc[key][4][-1] // 16

    # ---- per-layer geometry from layer 0 headers, uniformity asserted while writing ----
    Klay, stride = {}, {}
    for l in layers:
        K = K_of(f"model.language_model.layers.{l}.mlp.experts.0.gate_proj.trellis")
        assert K in (2, 3), f"layer {l} K {K}"
        for r in ROLES:
            tk = f"model.language_model.layers.{l}.mlp.experts.0.{r}.trellis"
            assert K_of(tk) == K, f"L{l} {r} K differs"
            assert s.nbytes(tk) == 204_800 * K, f"L{l} {r} trellis {s.nbytes(tk)}"
        Klay[l] = K
        stride[l] = align_up(614_400 * K + 19_216)  # the +16 tail word (31-B §1)

    # ---- experts.bin ----
    off = 0
    with open(out / "experts.bin", "wb") as fh:
        for l in layers:
            for e in range(N_EXPERT):
                pre = f"model.language_model.layers.{l}.mlp.experts.{e}."
                blob = bytearray()
                mul1s = []
                for r in ROLES:
                    for part in ("trellis", "suh", "svh"):
                        blob += s.raw(pre + r + "." + part)
                    b = s.raw(pre + r + ".mul1")
                    dt, shp = s.loc[pre + r + ".mul1"][3:5]
                    assert dt == "I32" and shp == () and len(b) == 4, f"{pre}{r}.mul1 {dt} {shp}"
                    mul1s.append(struct.unpack("<i", b)[0])
                blob += struct.pack("<4i", *mul1s, 0)
                assert len(blob) == 614_400 * Klay[l] + 19_216, f"L{l} E{e} {len(blob)}"
                fh.write(bytes(blob) + b"\x00" * (stride[l] - len(blob)))
                off += stride[l]
            if l % 12 == 0 or l == layers[-1]:
                print(f"experts.bin: layer {l} written, {off / 2**30:.2f} GiB", flush=True)
    total_experts = off

    # ---- native_experts.txt v6 (written after its data, before the manifest) ----
    with open(out / "native_experts.txt", "w") as fh:
        fh.write("# strata native experts v6: layer gu_type d_type offset blob_bytes K "
                 "(exl3; 200 = kTypeEXL3; offsets in this pack's experts.bin; raw blob "
                 f"= 614,400*K + 19,216 padded to {ALIGN} B; n_expert {N_EXPERT}, total {total_experts})\n")
        off = 0
        for l in layers:
            fh.write(f"{l} {K_TYPE_EXL3} {K_TYPE_EXL3} {off} {stride[l]} {Klay[l]}\n")
            off += stride[l] * N_EXPERT
        assert off == total_experts

    # ---- dense.bin + exl3_dense.txt + index.txt (C6.0: canonical names + engine forms, 46) ----
    grouped: dict[str, dict] = {}
    for k in densekeys:
        for suf in (".trellis", ".suh", ".svh", ".mul1"):
            if k.endswith(suf):
                grouped.setdefault(k[: -len(suf)], {})[suf[1:]] = k
                break
        else:
            grouped[k] = {}
    side, idxrows = [], []
    pool = 0
    with open(out / "dense.bin", "wb") as fh:
        def wr(b: bytes) -> int:
            nonlocal pool
            o = pool
            fh.write(b + b"\x00" * (align_up(len(b)) - len(b)))
            pool += align_up(len(b))
            return o

        for base in sorted(grouped):
            parts = grouped[base]
            cname, form = canon(base)
            if not parts:  # plain tensor: written in its engine FORM (46 §4.3)
                b = s.raw(base)
                dt, shp = s.loc[base][3:5]
                if form == "shape":  # token_embd: embd.gguf carries it; iq3_s-style shape-only row
                    idxrows.append((cname, 0, 0, 0, pool, 0, 2560, 248320, 8, 32))
                    continue
                b, form = convert_form(b, dt, form)
                off_ = wr(b)
                if cname.endswith(("ssm_conv1d.weight", "ple_conv1d.weight")):
                    assert len(shp) == 3 and shp[1] == 1 and shp[2] == 4, f"{base}: conv {shp}"
                    ne0, ne1 = shp[2], shp[0]                 # engine shape (4, 10240), 46
                else:
                    ne0, ne1 = shp[-1], shp[-2] if len(shp) > 1 else 0
                if cname.endswith("ffn_gate_inp_shexp.weight"):
                    ne1 = 0                                    # canonical 1-D form (46)
                idxrows.append((cname, KIND[form], off_, len(b), off_, len(b), ne0, ne1, 0, 1))
                continue
            assert set(parts) == {"trellis", "suh", "svh", "mul1"}, f"{base}: parts {set(parts)}"
            K = K_of(parts["trellis"])
            ki, nj, words = s.loc[parts["trellis"]][4]
            assert words == 16 * K, f"{base}: words {words} K {K}"
            mul1 = struct.unpack("<i", s.raw(parts["mul1"]))[0]
            if form == "exl3-split":
                # fused index_qk_proj (2560 -> 640 out) splits into indexer.q_proj (512) +
                # indexer.k_proj (128): trellis tile COLUMNS split 32+8 (each piece stands alone,
                # ld = its own n/16); suh duplicated; svh split (46 §2.11).
                assert ki * 16 == 2560 and nj * 16 == 640, f"{base}: indexer {ki}x{nj}"
                tr = np.frombuffer(s.raw(parts["trellis"]), "<u2").reshape(ki, nj, words)
                suh, svh = s.raw(parts["suh"]), s.raw(parts["svh"])
                assert len(suh) == 2 * ki * 16 and len(svh) == 2 * nj * 16
                t = wr(tr[:, :32, :].tobytes()); sq = wr(suh); vq = wr(svh[:2 * 512])
                side.append(f"{cname}.q_proj.weight {ki * 16} 512 {K} 2 {t} {sq} {vq} {mul1}")
                t = wr(tr[:, 32:, :].tobytes()); sk = wr(suh); vk = wr(svh[2 * 512:])
                side.append(f"{cname}.k_proj.weight {ki * 16} 128 {K} 2 {t} {sk} {vk} {mul1}")
                continue
            offs = {}
            for p in ("trellis", "suh", "svh"):
                offs[p] = wr(s.raw(parts[p]))
            side.append(f"{cname} {ki * 16} {nj * 16} {K} 2 {offs['trellis']} {offs['suh']} "
                        f"{offs['svh']} {mul1}")
        fh.truncate(pool)
    assert len(side) == 313 - 12 + 24, f"exl3_dense rows {len(side)} (46: 313 - 12 fused + 24 split)"
    assert len(idxrows) == 754, f"index rows {len(idxrows)}"
    assert len(side) + len(idxrows) == 1079, "46: vocab count must match the iq3_s ground truth"
    with open(out / "exl3_dense.txt", "w") as fh:
        fh.write("# strata exl3 dense v1: name ne0 ne1 K cb trellis_off suh_off svh_off mul1 "
                 f"(offsets in dense.bin; cb 2 = mul1 codebook; align {ALIGN}; canonical names per 46)\n")
        fh.write(f"# tensors {len(side)}\n")
        fh.write("\n".join(side) + "\n")
    with open(out / "index.txt", "w") as fh:
        fh.write(f"# strata pack index v3 -- generated by tools/exl3/pack_exl3.py from {src.name} "
                 f"(canonical vocabulary, 46; exl3 dense linears ride exl3_dense.txt, not rows here)\n")
        fh.write(f"# align {ALIGN} pool {pool} tensors {len(idxrows)}\n")
        for (name, kind, so, sb, do, db, ne0, ne1, cbits, g) in idxrows:
            fh.write(f"{name} 0 {kind} {so} {sb} {do} {db} {ne0} {ne1} {cbits} 0 {g} 0 0 0 0 0 0 0\n")

    # ---- ngram copy (byte-for-byte; hybrid consumption, runtime mmap) ----
    shutil.copyfile(src / ngram_file, out / ngram_file)
    ng_meta = s.headers[ngram_file]["meta"] or {}
    assert ng_meta.get("format") == "exl3_ngram_trellis", f"ngram format {ng_meta}"

    # ---- metadata.gguf carrier ----
    n_tensors, n_kv = gguf_meta_copy(a.gguf_meta, out / "metadata.gguf")
    print(f"metadata.gguf: n_tensors {n_tensors} -> 0, KV {n_kv} records copied verbatim")

    # ---- manifest.json LAST (completion marker) ----
    man = {"tool": "tools/exl3/pack_exl3.py v2 (46: canonical vocab)", "src": src.name, "counts": cnt,
           "layers": len(layers), "n_expert": N_EXPERT, "align": ALIGN,
           "K_per_layer": {str(l): Klay[l] for l in layers},
           "blob_stride": {str(l): stride[l] for l in layers},
           "files": {}}
    for f in ("experts.bin", "native_experts.txt", "dense.bin", "exl3_dense.txt", "index.txt",
              ngram_file, "metadata.gguf", "embd.gguf"):
        p = out / f
        if not p.exists():
            continue
        man["files"][f] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    with open(out / "manifest.json", "w") as fh:
        json.dump(man, fh, indent=1)
    s.close()
    print(f"done: pack in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
