import re
import torch
from numogpt.bpe import get_encoder
from torch.utils.data import Dataset
from pathlib import Path


def load_stopwords(file_path: str):
    f = Path(file_path)
    if f.exists():
        return set([line.replace('\n', '') for line in open(str(f), 'r', encoding='utf-8').readlines()])
    return set()


# (wordacy-nn) original modified: #removed hypher
def str_tokenize_words(s: str, stopwords=set()):
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", str.lower(s))
    if words: return [w for w in words if w not in stopwords]
    return []

class TextFlattenDataset(Dataset):

    def build_dataset(self, tokens, block_size):
        X, Y = [], []
        for i in range(len(tokens) - block_size):
            X.append(tokens[i : i + block_size])
            Y.append(tokens[i + 1 : i + block_size + 1])
        return torch.tensor(X), torch.tensor(Y)


    def __init__(self, path_file: str, block_size: int, stopwords_path: str=None):

        self.block_size = block_size
        self.encoder = get_encoder()
        text = None
        stopwords = load_stopwords(stopwords_path) if stopwords_path else set()

        with open(path_file, "r", encoding="utf-8") as f:
            max_len = 0
            text = f.read()
            tokens_list = str_tokenize_words(text, stopwords)

            # determine maximum
            for i in range(len(tokens_list) - block_size):
                txt = " ".join(tokens_list[i : i + block_size])
                sz = len(self.encoder.encode(txt))
                max_len =  sz if sz > max_len else max_len
            print("max_len=", max_len)

            text = " ".join(tokens_list)


        tokens = self.encoder.encode(text)
        self.X, self.Y = self.build_dataset(tokens, block_size)
        assert(len(self.X) == len(self.Y))


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_vocab_size(self):
        return len(self.encoder.encoder.items())

    def get_block_size(self):
        return self.block_size
