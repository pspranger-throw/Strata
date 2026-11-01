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
    dlines = [l.split() for l in (pack / "exl3_dense.txt").read_text().splitlines()
              if l and not l.startswith("#")]
    check(len(dlines) == sum(1 for p_ in groups.values() if p_),
          f"exl3_dense rows == source linear groups ({len(dlines)})")
    n_bad = 0
    with open(pack / "dense.bin", "rb") as df:
        def region(off_: int, key: str) -> bytes:
            df.seek(int(off_))
            b = df.read(raw.size(key))
            return b
        for f_ in dlines:
            name, to, so, vo = f_[0], f_[5], f_[6], f_[7]
            for part, off_ in (("trellis", to), ("suh", so), ("svh", vo)):
                if region(off_, name + "." + part) != raw[name + "." + part]:
                    n_bad += 1
            if int(f_[8]) != struct.unpack("<i", raw[name + ".mul1"])[0]:
                n_bad += 1
    check(n_bad == 0, f"exl3_dense every region ({n_bad} bad fields)")
    ilines = [l.split() for l in (pack / "index.txt").read_text().splitlines()
              if l and not l.startswith("#")]
    n_bad = 0
    with open(pack / "dense.bin", "rb") as df:
        for f_ in ilines:
            name, off_, nb = f_[0], int(f_[3]), int(f_[4])
            df.seek(off_)
            if df.read(nb) != raw[name]:
                n_bad += 1
    check(n_bad == 0, f"index rows every byte ({n_bad} bad)")
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
            for f_ in dlines:
                name = f_[0]
                if not (name.startswith("lm_head") or name.endswith("shared_expert.gate_proj")):
                    continue
                w_src = exl3.reconstruct_weight(raw.np(name + ".trellis"), raw.np(name + ".suh"),
                                                raw.np(name + ".svh"), CB_MUL1)
                tr = Region(df, int(f_[5]), name + ".trellis").a
                su = Region(df, int(f_[6]), name + ".suh").a
                sv = Region(df, int(f_[7]), name + ".svh").a
                w_pk = exl3.reconstruct_weight(tr, su, sv, CB_MUL1)
                check(np.array_equal(w_src, w_pk), f"decode parity dense {name} K{f_[3]} bit-exact {w_src.shape}")

    print("FAILURES:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
