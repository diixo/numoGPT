from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=5000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"])


with open("data/data.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer.train_from_iterator([text], trainer)

tokenizer.model.save("tests", "my_bpe_tokenizer")

print("Training complete! The vocabulary and rules are preserved.")
