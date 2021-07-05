from secrets import token_bytes


def random_key(len: int) -> int:
    """Generates len random bytes"""
    tb: bytes = token_bytes(len)
    return int.from_bytes(tb, "big")


def encrypt(original: str) -> tuple[int, int]:
    original_bytes = original.encode()
    dummy_key = random_key(len(original_bytes))
    original_key = int.from_bytes(original_bytes, "big")
    encrypted = original_key ^ dummy_key
    return dummy_key, encrypted


def decrypt(key1: int, key2: int) -> str:
    decrypted = key1 ^ key2
    tmp = decrypted.to_bytes((decrypted.bit_length() + 7) // 8, "big")
    return tmp.decode()


if __name__ == "__main__":
    key1, key2 = encrypt("Paul Simon writes a mean record")
    result = decrypt(key1, key2)
    print(result)
