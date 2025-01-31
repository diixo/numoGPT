
from numogpt import bpe
from numogpt.bpe import BPETokenizer, Encoder



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
    test_gpt2()
