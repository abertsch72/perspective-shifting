# Perspective Shift

This repo contains the code for the Findings of EMNLP 2022 paper "He Said, She Said: Style Transfer for Shifting the Perspective of Dialogues"

## Reproducibility Information 

for reproducing extractive summarization results: refer to [PreSumm](https://github.com/nlpyang/PreSumm) for their implementation of their extractive summarization model.

### computing resources used
All models were trained on either a v100 32GB GPU or a 2080TI GPU, with batch sizes/gradient accumulation steps adjusted for the differing memory size to maintain the same effective batch size.

### runtimes
On a 2080TI GPU, all BART-large perspective shifting models trained in under 6 hours. On a v100 GPU, these models trained in less than 2 hours.
Extractive summarization experiments were run on a 2080TI GPU and trained in less than 30 minutes when early stopping was used with a batch size of 1500.

### number of parameters
The BART-large model contains approximately 406M parameters. The BERT-base-uncased models used for extractive summarization contain approximately 120M parameters.

