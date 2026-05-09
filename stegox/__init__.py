__all__ = ["BinsStego", "DiscopStego", "EditStego", "HuffmanStego", "NeuralStego"]


def __getattr__(name):
    if name == "BinsStego":
        from stegox.methods.bins import BinsStego

        return BinsStego
    if name == "DiscopStego":
        from stegox.methods.discop import DiscopStego

        return DiscopStego
    if name == "EditStego":
        from stegox.methods.edit import EditStego

        return EditStego
    if name == "HuffmanStego":
        from stegox.methods.huffman_stego import HuffmanStego

        return HuffmanStego
    if name == "NeuralStego":
        from stegox.methods.neural import NeuralStego

        return NeuralStego
    raise AttributeError(name)
