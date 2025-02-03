import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from numogpt.utils import set_seed, evaluate_gpt
from numogpt.model import GPT
from numogpt.bpe import BPETokenizer, get_encoder
from numogpt.trainer import Trainer
from numogpt.text_dataset import TextFlattenDataset
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path


set_seed(3407)

context_sz = 8
max_iters = 2000
model_type = "gpt-numo"
use_mingpt = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Generate maximum of 2*n_words tokens (to cut-off unnecessary ones)
    max_tokens = len(tokens) + 2 * n_words
    generated_tokens = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, do_sample=False)

    generated_text = dataset.encoder.decode(generated_tokens.squeeze().tolist())

    cleaned_text = generated_text.replace("<|endoftext|>", "").replace("\n", " ")

    words = cleaned_text.split()
    trimmed_text = " ".join(words[:n_words])

    print('-' * 80)
    print(trimmed_text)
    print('-' * 80)


@torch.no_grad()
def predict_next(model: GPT, dataset: TextFlattenDataset, prompt: str):
    model.eval()

    tokens = dataset.encoder.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    idx_next = model.predict_next_word(idx, top_k=10)

    #next_word = dataset.encoder.decode(next_id.squeeze().tolist())
    next_word = dataset.encoder.decode([idx_next.item()])

    print('-' * 80)
    print(f"predict_next_word({prompt}:{next_word})")
    print('-' * 80)


@torch.no_grad()
def predict_next_word(model: GPT, dataset: TextFlattenDataset, word: str, device="cpu"):
    model.eval()

    tokens = dataset.encoder.encode(word)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    logits, _ = model(idx)
    logits = logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probs, dim=-1).item()

    next_word = dataset.encoder.decode([predicted_id])
    print('-' * 80)
    print(f"predict_next_word({word}):{next_word}")
    print('-' * 80)


def main():

    text_dataset = TextFlattenDataset("data/train-nn.txt", block_size=context_sz)

    gpt_config = GPT.get_default_config()
    gpt_config.model_type = model_type
    gpt_config.vocab_size = text_dataset.get_vocab_size()
    gpt_config.block_size = text_dataset.get_block_size()

    model = GPT(gpt_config)

    model_path = f"models/model-{context_sz}-{gpt_config.n_layer}-{gpt_config.n_head}-{gpt_config.n_embd}-{int(max_iters/1000)}k.pth"

    #---------------------------------------------------------------------------

    train_config = Trainer.get_default_config()
    train_config.device = "cpu"
    train_config.max_iters = max_iters
    train_config.batch_size = 32
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, text_dataset)

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
    else:
        trainer.run()
        print('-' * 80)

        torch.save(model.state_dict(), model_path)

    #---------------------------------------------------------------------------

    val_loss, ppl = evaluate_gpt(model, text_dataset, train_config.batch_size, train_config.device)
    print(f"val_loss={val_loss:.4f}, perplexity(PPL)={ppl:.4f}")

    #---------------------------------------------------------------------------

    #generate(model=gpt, prompt="text", num_samples=5)
    generate_n_words(model=model, dataset=text_dataset, prompt="text", n_words=10)
    #predict_next(model, text_dataset, "text")
    #predict_next_word(model, text_dataset, "text")



if __name__ == "__main__":
    main()
