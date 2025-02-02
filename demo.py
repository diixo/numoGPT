
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from numogpt.utils import set_seed
from numogpt.model import GPT
from numogpt.bpe import Encoder, get_encoder
from numogpt.bpe import BPETokenizer
from numogpt.trainer import Trainer
from keras.preprocessing.sequence import pad_sequences


set_seed(3407)


use_mingpt = True
model_type = "gpt-numo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextFlattenDataset(Dataset):

    def build_dataset(self, tokens, block_size):
        X, Y = [], []
        for i in range(len(tokens) - block_size):
            X.append(tokens[i : i + block_size])
            Y.append(tokens[i + 1 : i + block_size + 1])
        return torch.tensor(X), torch.tensor(Y)

    def __init__(self, text, block_size):
        self.block_size = block_size
        self.encoder = get_encoder()
        tokens = self.encoder.encode(text)
        self.X, self.Y = self.build_dataset(tokens, block_size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_vocab_size(self):
        return len(self.encoder.encoder.items())

    def get_block_size(self):
        return self.block_size



def generate(model: GPT, prompt="", num_samples=10, steps=20, do_sample=True):
        
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = tokenizer(prompt).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '':
            # to create unconditional samples...
            # huggingface/transformers tokenizer special cases these strings
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
        x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=50)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        #out = tokenizer.decode(y[i].tolist())
        print('-' * 80)
        print(out)



"""
def build_dataset_with_padding(texts, block_size):
    pad_token = 50256  # <|endoftext|>
    tokenized = [encoder.encode(text) for text in texts]

    X_padded = pad_sequences(tokenized, maxlen=block_size, padding="post", value=pad_token)

    X = torch.tensor(X_padded, dtype=torch.long)
    Y = torch.tensor(np.roll(X_padded, shift=-1, axis=1), dtype=torch.long)
    return X, Y
"""


def generate_text(model: GPT, text_dataset: TextFlattenDataset, prompt: str, max_tokens=50):
    model.eval()

    tokens = text_dataset.encoder.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)         # Get last token
            next_token = torch.multinomial(probs, num_samples=1)    # sampling

        x = torch.cat((x, next_token), dim=1)   # attach to final sequence

        if next_token.item() == text_dataset.encoder.encoder["<|endoftext|>"]:
            break

    return text_dataset.encoder.decode(x.squeeze().tolist())    # decoded to text


def generate_n_words(
        model: GPT,
        dataset: TextFlattenDataset,
        prompt: str,
        n_words=10,
        temperature=1.0,
        top_k=10
    ):
    model.eval()

    tokens = dataset.encoder.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate maximum of 3*n_words tokens (to cut-off unnecessary ones)
    max_tokens = len(tokens) + 2 * n_words
    generated_tokens = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, do_sample=False)

    generated_text = dataset.encoder.decode(generated_tokens.squeeze().tolist())

    cleaned_text = generated_text.replace("<|endoftext|>", "").replace("\n", " ")

    words = cleaned_text.split()
    trimmed_text = " ".join(words[:n_words])

    print('-' * 80)
    print(trimmed_text)
    print('-' * 80)



def main():
    text = None

    with open("data/train-nn.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_dataset = TextFlattenDataset(text, block_size=32)

    config = GPT.get_default_config()
    config.model_type = model_type
    config.vocab_size = text_dataset.get_vocab_size()
    config.block_size = text_dataset.get_block_size()

    gpt = GPT(config)

    train_config = Trainer.get_default_config()
    train_config.device = device
    train_config.max_iters = 1000
    train_config.batch_size = 32
    train_config.num_workers = 0
    trainer = Trainer(config=train_config, model=gpt, train_dataset=text_dataset)
    trainer.run()

    #generate(model=gpt, prompt="text", num_samples=5)
    generate_n_words(model=gpt, dataset=text_dataset, prompt="text", n_words=10)



if __name__ == "__main__":
    main()
