# numoGPT

A minimal PyTorch re-implementation of OpenAI GPT (Generative Pretrained Transformer) training [GPT](https://github.com/openai/gpt-2), both training and inference. numoGPT tries to be small, clean, interpretable and educational, as most of the currently available GPT model implementations can a bit sprawling. 
GPT is not a complicated model and this implementation is appropriately about 300 lines of code (see [numogpt/model.py](numogpt/model.py)). All that's going on is that a sequence of indices feeds into a [Transformer](https://arxiv.org/abs/1706.03762), and a probability distribution over the next index in the sequence comes out. 
The majority of the complexity is just being clever with batching (both across examples and over sequence length) for efficiency.

* **numoGPT** is alternative fork from [minGPT](https://github.com/karpathy/minGPT)


## Updates:
* [Demo](demo.py) has been implemented, that demonstrated training on input [train text](data/train-nn.txt).
* Embedded openai' GPT2: [tokens volabulary](gpt-2/vocab.bpe), [json-vocabulary](gpt-2/encoder.json) of indices for encoder.
* Implemented filtering by customized stopwords: [stopwords.txt](data/stopwords.txt).
* Implemented pytorch **TextDataset** ([text_dataset.py](numogpt/text_dataset.py)) with splitting the input text into token-blocks.


## Model:
Working demo: [demo.py](demo.py)

* device:  **cpu**
* model:   **gpt-numo**
* n_layer: **4**
* n_head:  **4**
* n_embd:  **64**
* block_sz:   **8**
* params:  **3.42M**
* **stopwords**


### tokens_block size=8:
```
TextWordacyDataset.sz=4677, block_size=6, blocks=780
number of parameters: 0.28M
running on device: cpu
...on 100th iter...
...on 200th iter...
...on 300th iter...
...on 400th iter...
...on 500th iter...
...
...on 4600th iter...
...on 4700th iter...
...on 4800th iter...
...on 4900th iter...
...on 5000th iter...
...finished 5000 iter(s)
--------------------------------------------------------------------------------
evaluate_gpt epoch:: batches=74, batch_sz=64
val_loss=0.5704, perplexity(PPL)=1.7690
--------------------------------------------------------------------------------
prompt ("text"): clustering algorithms sota pretrained language
--------------------------------------------------------------------------------
```

![numoGPT](assets/figure-1.png)

### Text generation by specified prompt:
```
--------------------------------------------------------------------------------
prompt ("text"): clustering models proposed consider
--------------------------------------------------------------------------------
```


### References:

* **1.** [minGPT](https://github.com/karpathy/minGPT) from Andrej Karpathy
* **2.**  [minbpe](https://github.com/karpathy/minbpe) from Andrej Karpathy
* **3.** [Karpathy-nn-zero-to-hero-gpt-exercises](https://www.kaggle.com/code/chizkidd/karpathy-nn-zero-to-hero-gpt-exercises/notebook)
* **4.** [Training a Mini-GPT to Learn Two-Digit Addition](https://www.gaohongnan.com/influential/generative_pretrained_transformer/05_adder.html)
