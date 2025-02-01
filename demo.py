
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


set_seed(3407)

use_mingpt = True
model_type = 'gpt-noomo'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():
    text = None

    with open("data/text-f-frq-5.txt", "r", encoding="utf-8") as f:
        text = f.read()


    enc = get_encoder()
    encoded_text = enc.encode(text)
    np_data = np.array(encoded_text, dtype=np.int32)
    data = torch.tensor(np_data, dtype=torch.long)

    config = GPT.get_default_config()
    config.model_type = "gpt-numo"
    config.vocab_size = len(enc.encoder.items())
    config.block_size = 64     #block_size = context_size
    #config.n_layer = 6
    #config.n_head = 8
    #config.n_embd = 256
    model = GPT(config)

    config = Trainer.get_default_config()
    config.device = device
    config.max_iters = None
    trainer = Trainer(config, model, data)
    trainer.run()



if __name__ == "__main__":
    main()
