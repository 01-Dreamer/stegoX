from stegox.core import bits_to_bytes, bytes_to_bits, chunk_bits
from stegox.methods.huffman import HuffmanCoding


def test_bytes_to_bits_round_trip_with_header():
    payload = b"hello"
    assert bits_to_bytes(bytes_to_bits(payload)) == payload


def test_chunk_bits_pads_last_block():
    assert chunk_bits("10101", 2) == ["10", "10", "10"]


def test_huffman_codes_cover_all_symbols():
    huffman = HuffmanCoding()
    huffman.make_heap_from_array([0.5, 0.25, 0.25])
    huffman.merge_nodes()
    huffman.make_codes()
    assert set(huffman.codes) == {0, 1, 2}
    assert all(code for code in huffman.codes.values())
