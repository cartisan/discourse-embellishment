# Discourse Embellishment
The LSTM Decoder-Encoder setup for the paper [Discourse Embellishment Using a Deep Encoder-Decoder Network](https://arxiv.org/abs/1810.08076), training to embellish text i.e. make more complicated sentence from less complicated ones while keeping the semantics.

## Setup
Your need to install [OpenNMT-tf:](https://github.com/OpenNMT/OpenNMT-tf) 
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
* [Tutorial used as reference and to valid setup (on en-vi translation dataset)](https://github.com/tensorflow/nmt)
