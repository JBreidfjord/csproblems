class CompressedGene:
    def __init__(self, gene: str) -> None:
        """Compresses a gene sequence into a bit string"""
        self._compress(gene)

    def _compress(self, gene: str) -> None:
        self.bit_string: int = 1  # sentinel
        for nucleotide in gene.upper():
            self.bit_string <<= 2  # shift left two bits
            if nucleotide == "A":
                self.bit_string |= 0b00
            elif nucleotide == "C":
                self.bit_string |= 0b01
            elif nucleotide == "G":
                self.bit_string |= 0b10
            elif nucleotide == "T":
                self.bit_string |= 0b11
            else:
                raise ValueError(f"Invalid nucleotide: {nucleotide}")

    def _decompress(self) -> str:
        gene: str = ""
        for i in range(0, self.bit_string.bit_length() - 1, 2):  # -1 excludes sentinel
            bits: int = self.bit_string >> i & 0b11  # reads two bits
            if bits == 0b00:
                gene += "A"
            elif bits == 0b01:
                gene += "C"
            elif bits == 0b10:
                gene += "G"
            elif bits == 0b11:
                gene += "T"
            else:
                raise ValueError(f"Invalid bits: {bits}")
        return gene[::-1]

    def __str__(self) -> str:
        return self._decompress()


if __name__ == "__main__":
    from sys import getsizeof

    original = "TAGGGATTAACCGTTATATATATATAGCCATGGATCGATTATA" * 100
    print(f"Original data is {getsizeof(original)} bytes")
    compressed = CompressedGene(original)
    print(f"Compressed data is {getsizeof(compressed)} bytes")
    print(
        f"Original and decompressed data are the same size: {original == str(compressed)}"
    )
