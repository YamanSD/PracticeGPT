from Tokenizer import decode, encode


def main() -> None:
    """
    Main function of the program.
    :return:
    """

    print(encode("hii there"))
    print(decode(encode("hii there")))
    return


if __name__ == "__main__":
    main()
