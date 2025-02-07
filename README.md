# numoGPT

* Alternative repository of minGPT, implemented by Andrej Karpathy.
* Demo has been implemented additionally, that demonstrated training on an input text file.
* Embedded openai's: tokens volabulary, json-vocabulary indices of encoder


## Model:
Working demo: [demo.py](demo.py)

* device:  **cpu**
* model:   **gpt-numo**
* n_layer: **4**
* n_head:  **4**
* n_embd:  **64**
* block:   **8**
* params:  **3.42M**


### context-size~8:
```
number of parameters: 3.42M
running on device: cpu
...on 100th iter...
...on 200th iter...
...on 300th iter...
...on 400th iter...
...on 500th iter...
...
...on 1600th iter...
...on 1700th iter...
...on 1800th iter...
...on 1900th iter...
...on 2000th iter...
...finished 2000 iter(s)
--------------------------------------------------------------------------------
evaluate_gpt:: sz=40, batch_sz=32
val_loss=1.4798, perplexity(PPL)=4.3919
--------------------------------------------------------------------------------
text and the clustering results
--------------------------------------------------------------------------------
```



### Embedded:
* encoding: **gpt2** (124M)
* vocabular: **gpt2** (124M)


### References:

* **1.** [minGPT](https://github.com/karpathy/minGPT) from Andrej Karpathy
* **2**  [minbpe][https://github.com/karpathy/minbpe] from Andrej Karpathy
* **3.** [Karpathy-nn-zero-to-hero-gpt-exercises](https://www.kaggle.com/code/chizkidd/karpathy-nn-zero-to-hero-gpt-exercises/notebook)
* **4.** [Training a Mini-GPT to Learn Two-Digit Addition](https://www.gaohongnan.com/influential/generative_pretrained_transformer/05_adder.html)
