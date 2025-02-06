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

    def build_dataset_fixed_blocks(self, text, block_size):
        pad_token_id = 50256 #encode("<|endoftext|>")
        words = self.encoder.pre_tokenize(text)

        current_tokens = []     # for idxs
        current_wtokens = []    # for words

        blocks = []             # for idxs
        wblocks = []            # for words

        for word in words:
            word_tokens = self.encoder.encode(word)

            if len(current_tokens) + len(word_tokens) > block_size:
                if pad_token_id is not None:
                    current_tokens += [pad_token_id] * (block_size - len(current_tokens))
                blocks.append(current_tokens)
                current_tokens = []
                current_wtokens = []
            current_tokens.extend(word_tokens)
            current_wtokens.append(word)

        # fill last block if it exist
        if current_tokens:
            if pad_token_id is not None:
                current_tokens += [pad_token_id] * (block_size - len(current_tokens))
            blocks.append(current_tokens)
            wblocks.append(current_wtokens)

        return blocks


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
            text = f.read()
            tokens_list = str_tokenize_words(text, stopwords)
            text = " ".join(tokens_list)

        #self.build_dataset_fixed_blocks(text, block_size)
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
