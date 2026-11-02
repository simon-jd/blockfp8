"""The minimal Parquet reader behind the wikitext-2 perplexity run (no GPU)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import wikitext  # noqa: E402


def test_snappy_literal_and_overlapping_copy():
    # 12 bytes: literal "abc", then a 9-byte copy from 3 back (overlaps itself).
    assert wikitext.snappy_decompress(bytes([12, 0x08]) + b"abc" + bytes([0x15, 0x03])) == b"abc" * 4


def test_snappy_rejects_wrong_length():
    with pytest.raises(ValueError):
        wikitext.snappy_decompress(bytes([13, 0x08]) + b"abc" + bytes([0x15, 0x03]))


def test_rle_hybrid_bit_packed_then_run():
    # One bit-packed group of 8 one-bit values (LSB first), then a run of ten 1s.
    data = bytes([0x03, 0x8D, 0x14, 0x01])
    assert wikitext.rle_hybrid(data, 0, len(data), 1, 18) == [1, 0, 1, 1, 0, 0, 0, 1] + [1] * 10


@pytest.mark.skipif(not os.path.exists(wikitext.PATH), reason="wikitext-2 test parquet not downloaded")
def test_reads_the_test_split():
    rows, digest = wikitext.load_test()
    assert digest == wikitext.SHA256 and len(rows) == wikitext.ROWS
    assert all(isinstance(r, str) for r in rows)
    assert rows[0] == "" and rows[1] == " = Robert Boulter = \n"
