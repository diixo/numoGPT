
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from numogpt.utils import set_seed
from numogpt.model import GPT
from numogpt.bpe import Encoder, get_encoder
from numogpt.bpe import BPETokenizer
from numogpt.trainer import Trainer
from keras.preprocessing.sequence import pad_sequences


set_seed(3407)

use_mingpt = True
model_type = 'gpt-noomo'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = get_encoder()


def generate(model, prompt='', num_samples=10, steps=20, do_sample=True):
        
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
        print('-'*80)
        print(out)


"""
def generate_small():
    bpe_tokenizer = BPETokenizer()
    context = torch.tensor([[bpe_tokenizer.encoder['<|endoftext|>']]], dtype=torch.long)
    generated_text = model.generate(context, max_new_tokens=100)

    decoded = bpe_tokenizer.decode(generated_text[0].tolist())
    print(decoded)  # Вывод сгенерированного текста
"""


def build_dataset(tokens, block_size):
    X, Y = [], []
    for i in range(len(tokens) - block_size):
        X.append(tokens[i : i + block_size])
        Y.append(tokens[i + 1 : i + block_size + 1])
    return torch.tensor(X), torch.tensor(Y)


def build_dataset_with_padding(texts, block_size):
    pad_token = 50256  # <|endoftext|>
    tokenized = [encoder.encode(text) for text in texts]

    X_padded = pad_sequences(tokenized, maxlen=block_size, padding="post", value=pad_token)

    X = torch.tensor(X_padded, dtype=torch.long)
    Y = torch.tensor(np.roll(X_padded, shift=-1, axis=1), dtype=torch.long)
    return X, Y


def main():
    text = None

    with open("data/train-nn.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = encoder.encode(text)
    #np_data = np.array(encoded_text, dtype=np.int32)
    #data = torch.tensor(np_data, dtype=torch.long)

    config = GPT.get_default_config()
    config.model_type = "gpt-numo"
    config.vocab_size = len(encoder.encoder.items())
    config.block_size = 32     #block_size = context_size
    #config.n_layer = 6
    #config.n_head = 8
    #config.n_embd = 256
    model = GPT(config)

    X_train, Y_train = build_dataset(tokens, config.block_size)

    train_config = Trainer.get_default_config()
    train_config.device = device
    train_config.max_iters = None
    train_config.batch_size = 32
    train_config.num_workers = 0
    trainer = Trainer(config=train_config, model=model, train_dataset=(X_train, Y_train))
    trainer.run()
    print("...finished.")



if __name__ == "__main__":
    main()
