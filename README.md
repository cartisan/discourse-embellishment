# Discourse Embellishment
The LSTM Encoder-Decore setup used in the paper [Discourse Embellishment Using a Deep Encoder-Decoder Network](https://arxiv.org/abs/1810.08076).

## Abstract
We suggest a new NLG task in the context of the discourse generation pipeline of computational storytelling systems. This task, textual embellishment, is defined by taking a text as input and generating a semantically equivalent output with increased lexical and syntactic complexity. Ideally, this would allow the authors of computational storytellers to implement just lightweight NLG systems and use a domain-independent embellishment module to translate its output into more literary text. We present promising first results on this task using LSTM Encoder-Decoder networks trained on the Wiki-Large dataset.

## Setup
You need to install [OpenNMT-tf:](https://github.com/OpenNMT/OpenNMT-tf) 
`pip install -r requirements.txt`

## Execution
Adopt the paths in `config/data.yml` to point according to your setup.
Especially, download and point the config to pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) word vector embeddings.
To train a model, from the root dir of this project execute:
`onmt-main train_and_eval --model config/nmt_medium.py --config config/opennmt-wikilarge.yml config/data.yml`

To manually evaluate the trained model and inspect its results, from the root dir of this project execute:
`onmt-main infer --model config/nmt_medium.py --config config/opennmt-defaults.yml config/data.yml --features_file data/wikilarge.valid.simple > results/wikilarge.valid.gen`
Using the BLEU-calculation script provided by OpenNMT allows to evaluate the generated text. From the dir where OpenNMT is installed, execute: `perl OpenNMT-tf/third_party/multi-bleu.perl complexicator/data/wikilarge.valid.complex < wikilarge.valid.gen`

## Further References
* [Sentence Simplification with Deep Reinforcement Learning (Zhang et al. 2017)](http://aclweb.org/anthology/D/D17/D17-1062.pdf) (Some fancy enc/dec based performance hacks)
* [An Experimental Study of LSTM Encoder-Decoder Model for Text Simplification (Wang et al. 2016)](https://arxiv.org/pdf/1609.03663.pdf) (Ways of evaluating structural changes learned by enc/dec)
* [Tutorial used as reference and to validate setup (on en-vi translation dataset)](https://github.com/tensorflow/nmt)

## Citation
```
@inproceedings{berov_standvoss_2018,
address = {Tilburg, The Netherlands},
title = {Discourse Embellishment Using a Deep Encoder-Decoder Network},
booktitle = {3rd Workshop on Computational Creativity and Natural Language Generation},
publisher = {{ACL Antology}},
author = {Berov, Leonid and Standvoss, Kai},
year = {2018},
pages = {11--16}
}
```
