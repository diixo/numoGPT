
from numogpt import bpe
from numogpt.bpe import BPETokenizer, Encoder


def test_tokenizer():
    with open("data/data.txt", "r", encoding="utf-8") as f:
        text = f.read()
        text = text.lower()

    e = bpe.get_encoder()
    idxs = e.encode(text)
    print(idxs)

    tokens = e.pre_tokenize(text)

    text = "".join(tokens)
    idxs = e.encode(text)

    print(idxs)
    print(text)
    print(str(len(tokens)) + ",", len(idxs))



def test_gpt2():
    """ test gpt-2 models encoder on 50257 """

    with open("data/data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    e = bpe.get_encoder()
    r = e.encode_and_show_work(text)

    print(r['tokens'])

    for part in r['parts']:
        print(part)


def main():
    test_gpt2()


if __name__ == "__main__":
    print("Hello AI!")
    test_tokenizer()
