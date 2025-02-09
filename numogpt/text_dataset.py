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
    words = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", s)
    if words: return [w for w in words if w not in stopwords]
    return []

class TextDataset(Dataset):

    def build_dataset_words(self, text, block_size):
        pad_token_id = 50256 #encode("<|endoftext|>")
        words = self.encoder.pre_tokenize(text)

        current_tokens = []
        blocks_X = []
        blocks_Y = []

        i = 0
        stat = dict()

        for word in words:
            word_tokens = self.encoder.encode(word)

            if len(current_tokens) + len(word_tokens) > (block_size-1):
                if pad_token_id is not None:
                    current_tokens += [pad_token_id] * (block_size - len(current_tokens))
                blocks_X.append(current_tokens)

                # fill next_tokens with shifting from current_tokens[]
                next_tokens = current_tokens[1:]
                next_tokens.append(pad_token_id)
                blocks_Y.append(next_tokens)

                stat[i] = stat.get(i, 0) + 1
                i = 0
                current_tokens = []

            i += 1
            current_tokens.extend(word_tokens)

        # fill last block if it exist
        if current_tokens:
            if pad_token_id is not None:
                current_tokens += [pad_token_id] * (block_size - len(current_tokens))
            blocks_X.append(current_tokens)

            # fill next_tokens with shifting from current_tokens[]
            next_tokens = current_tokens[1:]
            next_tokens.append(pad_token_id)
            blocks_Y.append(next_tokens)

        assert(len(blocks_X) == len(blocks_Y))
        print("tokens/block distribution:", stat)
        return torch.tensor(blocks_X), torch.tensor(blocks_Y)


    def __init__(self, path_file: str, block_size: int, stopwords_path: str=None):

        self.block_size = block_size
        self.encoder = get_encoder()
        text = None
        stopwords = load_stopwords(stopwords_path) if stopwords_path else set()

        with open(path_file, "r", encoding="utf-8") as f:
            text = f.read()
            tokens_list = str_tokenize_words(text, stopwords)
            text = " ".join(tokens_list)

        self.X, self.Y = self.build_dataset_words(text, block_size)
        assert(len(self.X) == len(self.Y))
        #print(f"TextDataset.sz={len(self.X)}, block_size={block_size}, blocks={int(len(self.X)/block_size+1)}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_vocab_size(self):
        return len(self.encoder.encoder.items())

    def get_block_size(self):
        return self.block_size


class WordacyEncoder:

    def __init__(self):
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.stopwords = set()
        self.PAD_TOKEN = "[UNK]"


    def build_vocab(self, text: str, stopwords=set()):
        tokens = str_tokenize_words(text, set())
        self.word_to_id = {token: id for id, token in enumerate(dict.fromkeys(tokens))}
        self.id_to_word = {id: token for token, id in self.word_to_id.items()}
        self.stopwords = stopwords

        self.word_to_id[self.PAD_TOKEN] = self.word_to_id.get(self.PAD_TOKEN, len(self.word_to_id))
        self.id_to_word[self.word_to_id[self.PAD_TOKEN]] = self.PAD_TOKEN
        assert(len(self.word_to_id) == len(self.id_to_word))


    def encode(self, text: str):
        tokens = str_tokenize_words(text, self.stopwords)
        return [ self.word_to_id.get(token, self.word_to_id[self.PAD_TOKEN]) for token in tokens ]


    def decode(self, ids: list):
        words = [ self.id_to_word.get(id, self.id_to_word[self.word_to_id[self.PAD_TOKEN]]) for id in ids ]
        return " ".join(words)


class TextWordacyDataset(Dataset):

    def build_dataset(self, tokens, block_size):
        X, Y = [], []
        for i in range(len(tokens) - block_size):
            X.append(tokens[i : i + block_size])
            Y.append(tokens[i + 1 : i + block_size + 1])
        return torch.tensor(X), torch.tensor(Y)


    def __init__(self, path_file: str, block_size: int, stopwords_path: str=None):

        self.block_size = block_size
        self.encoder = WordacyEncoder()

        with open(path_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
            stopwords = load_stopwords(stopwords_path) if stopwords_path else set()
            self.encoder.build_vocab(text, stopwords)

        tokens = self.encoder.encode(text)
        self.X, self.Y = self.build_dataset(tokens, block_size)
        assert(len(self.X) == len(self.Y))
        print(f"TextWordacyDataset.sz={len(self.X)}, block_size={block_size}, blocks={int(len(self.X)/block_size+1)}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_vocab_size(self, n: int = 8):
        return (len(self.encoder.word_to_id) + n-1) // n * n

    def get_block_size(self):
        return self.block_size
