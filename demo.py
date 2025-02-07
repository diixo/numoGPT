import torch
from numogpt.utils import set_seed, evaluate_gpt
from numogpt.model import GPT
from numogpt.trainer import Trainer
from numogpt.text_dataset import TextDataset
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path


set_seed(3407)

tokens_block = 8
max_iters = 2000
model_type = "gpt-numo"
use_mingpt = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
def build_dataset_with_padding(texts, block_size):
    pad_token = 50256  # <|endoftext|>
    tokenized = [encoder.encode(text) for text in texts]

    X_padded = pad_sequences(tokenized, maxlen=block_size, padding="post", value=pad_token)

    X = torch.tensor(X_padded, dtype=torch.long)
    Y = torch.tensor(np.roll(X_padded, shift=-1, axis=1), dtype=torch.long)
    return X, Y
"""


def generate_text(model: GPT, text_dataset: TextDataset, prompt: str, max_tokens=50):
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
        dataset: TextDataset,
        prompt: str,
        n_words=10,
        temperature=1.0,
        top_k=10
    ):
    model.eval()

    tokens = dataset.encoder.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated_tokens = model.generate(x, max_new_tokens=n_words, temperature=temperature, top_k=top_k, do_sample=False)

    generated_text = dataset.encoder.decode(generated_tokens.squeeze().tolist())

    cleaned_text = generated_text.replace("<|endoftext|>", "").replace("\n", " ")

    words = cleaned_text.split()
    trimmed_text = " ".join(words[:n_words])

    print('-' * 80)
    print(trimmed_text)
    print('-' * 80)



@torch.no_grad()
def predict_next_word(model: GPT, dataset: TextDataset, word: str, device="cpu"):
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

    text_dataset = TextDataset("data/train-nn.txt", tokens_block, "data/stopwords.txt")

    gpt_config = GPT.get_default_config()
    gpt_config.model_type = model_type
    gpt_config.vocab_size = text_dataset.get_vocab_size()
    gpt_config.block_size = text_dataset.get_block_size()

    model = GPT(gpt_config)

    model_path = f"models/model-{tokens_block}-{gpt_config.n_layer}-{gpt_config.n_head}-{gpt_config.n_embd}-{int(max_iters/1000)}k.pth"

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
    print('-' * 80)
    val_loss, ppl = evaluate_gpt(model, text_dataset, train_config.batch_size, train_config.device)
    print(f"val_loss={val_loss:.4f}, perplexity(PPL)={ppl:.4f}")

    #---------------------------------------------------------------------------

    #generate(model=gpt, prompt="text", num_samples=5)
    generate_n_words(model=model, dataset=text_dataset, prompt="text", n_words=5)
    #predict_next_word(model, text_dataset, "text")



if __name__ == "__main__":
    main()
