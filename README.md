# PracticeGPT

A tiny LLM based on GPT-2.

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Overview

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [nanoGPT](https://github.com/karpathy/NanoGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![loss](./screenshots/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).


## Installation

Install the requirements using the following command:
```
pip install -r requirements.txt
```

## Acknowledgement

This implementation is logically identical to [NanoGPT](https://github.com/karpathy/nanoGPT).

The purpose of this project is to apply and expand my knowledge about LLMs.
