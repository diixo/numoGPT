
import torch
from numogpt.bpe import get_encoder
from torch.utils.data import Dataset


class TextFlattenDataset(Dataset):

    def build_dataset(self, tokens, block_size):
        X, Y = [], []
        for i in range(len(tokens) - block_size):
            X.append(tokens[i : i + block_size])
            Y.append(tokens[i + 1 : i + block_size + 1])
        return torch.tensor(X), torch.tensor(Y)


    def __init__(self, path_file: str, block_size: int):
        text = None
        with open(path_file, "r", encoding="utf-8") as f:
            text = f.read()

        self.block_size = block_size
        self.encoder = get_encoder()
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
