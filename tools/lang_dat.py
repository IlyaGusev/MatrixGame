#!/usr/bin/env python3
"""Parse Space Rangers 2 .dat files (Lang.dat, Main.dat). Stdlib only.

Format: [8-byte signature] u32 crc32 of plaintext, i32 ciphered seed,
then a rand31pm-XOR-ciphered ZL01 block (u32 size + zlib stream).
Plaintext is a tree: u8 type, UTF-16LE zero-terminated name;
type 1 (par): value wstr; type 2 (block): u8 sorted, u32 child count,
children (each prefixed by two u32 sort indices when sorted).

Usage:
  lang_dat.py Data/Lang.dat                 # dump tree as text to stdout
  lang_dat.py Data/Lang.dat -o Lang.txt
  lang_dat.py Data/Lang.dat --maps rust_port/assets/menu/maps.txt
"""

import argparse
import struct
import sys
import zlib

ENCRYPTION_KEYS = {
    "SR1": 0,
    "ReloadMain": 1050086386,
    "ReloadCache": 1929242201,
    "HDMain": -1310144887,
    "HDCache": -359710921,
}
SIGN_KEY_1 = 0xC83FCBF3
SIGN_KEY_2 = 0x7DB6C99D


def is_signed(data):
    if len(data) < 8:
        return False
    body = data[8:]
    d0 = len(body) ^ SIGN_KEY_1 ^ SIGN_KEY_2
    d4 = (
        zlib.crc32((zlib.crc32(body) ^ SIGN_KEY_2).to_bytes(4, "little") + body)
        ^ SIGN_KEY_1
    )
    return data[:8] == d0.to_bytes(4, "little") + d4.to_bytes(4, "little")


def xor_decrypt(data, seed):
    out = bytearray(len(data))
    state = seed
    for i, b in enumerate(data):
        hi, lo = divmod(state, 0x1F31D)
        state = lo * 0x41A7 - hi * 0xB14
        if state < 1:
            state += 0x7FFFFFFF
        out[i] = b ^ ((state - 1) & 0xFF)
    return bytes(out)


def decrypt(data):
    """Returns (fmt_name, plaintext ZL01 block)."""
    if is_signed(data):
        data = data[8:]
    content_hash, seed = struct.unpack_from("<Ii", data)
    payload = data[8:]
    for fmt, key in ENCRYPTION_KEYS.items():
        if xor_decrypt(payload[:4], seed ^ key) != b"ZL01":
            continue
        plain = xor_decrypt(payload, seed ^ key)
        if zlib.crc32(plain) != content_hash:
            raise ValueError(f"crc mismatch for key {fmt}")
        return fmt, plain
    raise ValueError("no known encryption key fits (not an SR2 .dat?)")


def unzip_zl01(block):
    assert block[:4] == b"ZL01", block[:4]
    size = struct.unpack_from("<I", block, 4)[0]
    out = zlib.decompress(block[8:])
    assert len(out) == size
    return out


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u8(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def wstr(self):
        end = self.pos
        while self.data[end : end + 2] != b"\0\0":
            end += 2
        s = self.data[self.pos : end].decode("utf-16le")
        self.pos = end + 2
        return s


def parse_item(r, sorted_blocks):
    """-> (name, value) for pars, (name, [children]) for blocks."""
    typ = r.u8()
    name = r.wstr()
    if typ == 1:
        return name, r.wstr()
    if typ == 2:
        is_sorted = bool(r.u8()) if sorted_blocks else False
        count = r.u32()
        children = []
        for _ in range(count):
            if is_sorted:
                r.u32(), r.u32()
            children.append(parse_item(r, sorted_blocks))
        return name, children
    raise ValueError(f"bad item type {typ} at {r.pos}")


def parse(path):
    fmt, block = decrypt(open(path, "rb").read())
    tree = unzip_zl01(block)
    # root block: no leading type/name byte, starts at the sorted flag
    sorted_blocks = fmt in ("SR1", "ReloadMain", "HDMain")
    _, root = parse_item(Reader(b"\2\0\0" + tree), sorted_blocks)
    return fmt, root


def dump(children, indent=0):
    pad = "    " * indent
    out = []
    for name, val in children:
        if isinstance(val, str):
            out.append(f"{pad}{name}={val}")
        else:
            out.append(f"{pad}{name} {{")
            out.extend(dump(val, indent + 1))
            out.append(f"{pad}}}")
    return out


# ── maps.txt extraction (menu + in-game briefings) ──────────────────

# The SR2 shell substitutes these at runtime; the standalone port has
# no player/planet context, so bake in neutral replacements.
def clean(text):
    for tag in ("<clr>", "<clrEnd>"):
        text = text.replace(tag, "")
    text = text.replace("ейнджер <Player>", "ейнджер")  # "Рейнджер X," address
    text = text.replace("<Player>", "рейнджер")
    for tag in ("<Planet>", "<Star>", "<ToPlanet>"):  # apposition after the noun
        text = text.replace(" " + tag, "").replace(tag, "")
    text = text.replace("<Money>", "1000")
    return text.replace("\t", " ").strip()


def slug(s):
    out = "".join(c if c.isalnum() else "_" for c in s.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def category(fields):
    """original = the 2004 campaign maps (Group=0: Учебная, missions,
    Террон). addon = Group=1 Access<=9 (Резиденция Лякуши … Рудники).
    other = the remaining skirmish block (Access>=10, Аль-Кагул …
    Зенит). extra = demo / story-PB / service maps the menu hides."""
    group = fields.get("Group", [""])[0]
    if group == "0":
        return "original"
    if group == "1":
        return "addon" if int(fields.get("Access", ["99"])[0]) <= 9 else "other"
    return "extra"


def write_maps(root, out_path):
    robots_map = next(v for name, v in root if name == "RobotsMap")
    # Length 0..4 → the labels the quick-launch form shows in the
    # right column («Малая» … «Большая»), from the FormLoadRobot block.
    form = next((v for name, v in root if name == "FormLoadRobot"), [])
    form_pars = {n: v for n, v in form if isinstance(v, str)}
    length_labels = {str(i): form_pars.get(f"Length{i}", "") for i in range(5)}
    lines = []
    for _, entry in robots_map:
        if isinstance(entry, str):
            continue
        fields = {}
        for name, val in entry:
            if isinstance(val, str):
                fields.setdefault(name, []).append(val)
        stem = fields.get("Map", [""])[0].rsplit(".", 1)[0]
        if not stem:
            continue
        lines.append(f"map\t{slug(stem)}")
        lines.append(f"name\t{fields.get('Name', [''])[0]}")
        lines.append(f"cat\t{category(fields)}")
        lines.append(f"side\t{fields.get('Side', [''])[0]}")
        lines.append(f"dif\t{length_labels.get(fields.get('Length', [''])[0], '')}")
        for tag, field in (
            ("desc", "GovTextStart"),
            ("begin", "RobotsStart"),
            ("win", "RobotsWin"),
            ("loose", "RobotsLoss"),
        ):
            for text in fields.get(field, []):
                # The HD story-PB maps (Mansion/Sanatory/Asylum) ship
                # literal placeholders; drop them so consumers fall back.
                if not text.startswith("Плейсхолдер"):
                    lines.append(f"{tag}\t{clean(text)}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{out_path}: {len(robots_map)} maps", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dat", help="path to .dat file")
    ap.add_argument("-o", "--out", help="write text dump to file")
    ap.add_argument("--maps", help="write RobotsMap menu/briefing file (maps.txt)")
    args = ap.parse_args()

    fmt, root = parse(args.dat)
    print(f"format: {fmt}", file=sys.stderr)

    if args.maps:
        write_maps(root, args.maps)
        return
    text = "\n".join(dump(root)) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
