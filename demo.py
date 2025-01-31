
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from numogpt.utils import set_seed

set_seed(3407)

# create a GPT instance
from numogpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
#model_config.vocab_size = train_dataset.get_vocab_size()
#model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)


def main():
    print("Hello noomoai")

if __name__ == "__main__":
    main()
