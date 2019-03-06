import tensorflow as tf
import opennmt as onmt

# documentation:
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/inputters/text_inputter.py

def model():
    return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",  # see data.yml
          embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512),
      encoder=onmt.encoders.BidirectionalRNNEncoder(
          num_layers=2,
          num_units=512,
          reducer=onmt.layers.ConcatReducer(),
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.2,
          residual_connections=False),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=2,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.2,
          residual_connections=False))
