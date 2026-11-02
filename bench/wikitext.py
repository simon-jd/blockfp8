"""wikitext-2-raw-v1, test split, read from the Hugging Face parquet file
without pyarrow (not a vLLM dependency): just enough of Parquet (Thrift
compact footer, snappy, PLAIN / dictionary pages, RLE definition levels) for
the one string column of that file.

    mkdir -p data/wikitext-2-raw-v1 && curl -L -o data/wikitext-2-raw-v1/test-00000-of-00001.parquet \\
      https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet
"""

import hashlib
import os
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(REPO, "data", "wikitext-2-raw-v1", "test-00000-of-00001.parquet")
URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
       "wikitext-2-raw-v1/test-00000-of-00001.parquet")
SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"  # the Hub's LFS oid
ROWS = 4358


def _varint(b, i):
    v = s = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return v, i


def _zigzag(v):
    return (v >> 1) ^ -(v & 1)


def _thrift_value(b, i, t):
    if t in (1, 2):  # bool inside a struct: the type is the value
        return t == 1, i
    if t == 3:
        return struct.unpack_from("b", b, i)[0], i + 1
    if t in (4, 5, 6):
        v, i = _varint(b, i)
        return _zigzag(v), i
    if t == 7:
        return struct.unpack_from("<d", b, i)[0], i + 8
    if t == 8:
        n, i = _varint(b, i)
        return bytes(b[i:i + n]), i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = _varint(b, i)
        out = []
        for _ in range(n):
            if et in (1, 2):  # bool inside a list: one byte
                out.append(b[i] == 1)
                i += 1
            else:
                v, i = _thrift_value(b, i, et)
                out.append(v)
        return out, i
    if t == 11:
        n, i = _varint(b, i)
        out = {}
        if n:
            kt, vt = b[i] >> 4, b[i] & 0x0F
            i += 1
            for _ in range(n):
                k, i = _thrift_value(b, i, kt)
                out[k], i = _thrift_value(b, i, vt)
        return out, i
    if t == 12:
        return _thrift_struct(b, i)
    raise ValueError(f"thrift compact type {t}")


def _thrift_struct(b, i):
    """{field id: value}; nested structs are dicts too."""
    out, fid = {}, 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return out, i
        delta, t = h >> 4, h & 0x0F
        if delta:
            fid += delta
        else:
            v, i = _varint(b, i)
            fid = _zigzag(v)
        out[fid], i = _thrift_value(b, i, t)


def snappy_decompress(b):
    n, i = _varint(b, 0)
    out = bytearray()
    while i < len(b):
        tag = b[i]
        i += 1
        kind = tag & 3
        if kind == 0:  # literal
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(b[i:i + nb], "little")
                i += nb
            ln += 1
            out += b[i:i + ln]
            i += ln
            continue
        if kind == 1:
            ln, off = ((tag >> 2) & 7) + 4, ((tag >> 5) << 8) | b[i]
            i += 1
        elif kind == 2:
            ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 2], "little")
            i += 2
        else:
            ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 4], "little")
            i += 4
        if off == 0 or off > len(out):
            raise ValueError("snappy: bad copy offset")
        start = len(out) - off
        for j in range(ln):  # copies may overlap their own output
            out.append(out[start + j])
    if len(out) != n:
        raise ValueError(f"snappy: {len(out)} bytes, header says {n}")
    return bytes(out)


def rle_hybrid(b, i, end, bit_width, count):
    """Parquet's RLE / bit-packed hybrid encoding; returns count values."""
    out = []
    nbytes = (bit_width + 7) // 8
    while len(out) < count and i < end:
        h, i = _varint(b, i)
        if h & 1:  # bit-packed groups of 8 values
            groups = h >> 1
            raw = int.from_bytes(b[i:i + groups * bit_width], "little")
            i += groups * bit_width
            mask = (1 << bit_width) - 1
            out += [(raw >> (bit_width * j)) & mask for j in range(groups * 8)]
        else:
            out += [int.from_bytes(b[i:i + nbytes], "little")] * (h >> 1)
            i += nbytes
    if len(out) < count:
        raise ValueError("rle: ran out of data")
    return out[:count]


def _plain_strings(b, i, count):
    out = []
    for _ in range(count):
        (n,) = struct.unpack_from("<i", b, i)
        out.append(b[i + 4:i + 4 + n].decode("utf-8"))
        i += 4 + n
    return out, i


def read_strings(path):
    """All values of the file's single string column, in row order (None for nulls)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("not a parquet file")
    (flen,) = struct.unpack_from("<i", data, len(data) - 8)
    meta, _ = _thrift_struct(data, len(data) - 8 - flen)
    leaves = [s for s in meta[2] if 5 not in s]  # schema elements without children
    if len(leaves) != 1 or leaves[0].get(1) != 6:  # one BYTE_ARRAY column
        raise ValueError("expected one string column")
    max_def = 1 if leaves[0].get(3) == 1 else 0  # OPTIONAL -> definition levels
    values = []
    for rg in meta[4]:
        (chunk,) = rg[1]
        cm = chunk[3]
        codec = cm[4]
        if codec not in (0, 1):
            raise ValueError(f"unsupported codec {codec}")
        pos = min(o for o in (cm.get(11), cm[9]) if o)  # the dictionary page comes first
        end = pos + cm[7]
        dictionary, seen = None, 0
        while seen < cm[5] and pos < end:
            ph, pos = _thrift_struct(data, pos)
            page = data[pos:pos + ph[3]]
            pos += ph[3]
            ptype = ph[1]
            if ptype == 2:  # dictionary page
                raw = snappy_decompress(page) if codec == 1 else page
                dictionary, _ = _plain_strings(raw, 0, ph[7][1])
                continue
            if ptype == 0:  # data page v1: levels and values compressed together
                raw = snappy_decompress(page) if codec == 1 else page
                n, enc = ph[5][1], ph[5][2]
                j = 0
                defs = [max_def] * n
                if max_def:
                    (ln,) = struct.unpack_from("<i", raw, 0)
                    defs = rle_hybrid(raw, 4, 4 + ln, 1, n)
                    j = 4 + ln
            elif ptype == 3:  # data page v2: levels uncompressed, then values
                h2 = ph[8]
                n, enc, dl, rl = h2[1], h2[4], h2[5], h2[6]
                levels, body = page[:rl + dl], page[rl + dl:]
                defs = rle_hybrid(levels, rl, rl + dl, 1, n) if max_def else [max_def] * n
                raw = snappy_decompress(body) if codec == 1 and h2.get(7, True) else body
                j = 0
            else:
                raise ValueError(f"unsupported page type {ptype}")
            present = sum(d == max_def for d in defs)
            if enc == 0:
                vals, _ = _plain_strings(raw, j, present)
            elif enc in (2, 8):
                bw = raw[j]
                vals = [dictionary[x] for x in rle_hybrid(raw, j + 1, len(raw), bw, present)]
            else:
                raise ValueError(f"unsupported encoding {enc}")
            it = iter(vals)
            values += [next(it) if d == max_def else None for d in defs]
            seen += n
    if len(values) != meta[3]:
        raise ValueError(f"{len(values)} values for {meta[3]} rows")
    return values


def load_test(path=PATH):
    """The test split's text lines and the file's sha256 (checked against the Hub)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; download it from {URL}")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest}, expected {SHA256}")
    rows = read_strings(path)
    if len(rows) != ROWS:
        raise ValueError(f"{len(rows)} rows, expected {ROWS}")
    return rows, digest
