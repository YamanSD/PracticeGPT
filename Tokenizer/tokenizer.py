def data_str() -> str:
    """
    Reads the input.txt file and returns its content.
    :return:
    """
    with open('input.txt', 'r', encoding='utf-8') as file:
        return file.read()


"""
Reference to the data input string.
"""
text: str = data_str()

"""
List of characters used in the data string.
"""
chars: tuple[str, ...] = tuple(sorted(set(text)))
vocab_size: int = len(chars)

"""
Character to index and index to character mappings.
"""
stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
itos: dict[int, str] = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> list[int]:
    """
    Encodes the given string.
    :param s: string to be encoded.
    :return:
    """
    return [stoi[c] for c in s]


def decode(es: list[int]) -> str:
    """
    Decodes the given sequence of integers.
    :param es: sequence to be decoded.
    :return:
    """
    return ''.join(itos[i] for i in es)
