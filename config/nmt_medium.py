import tensorflow as tf
import opennmt as onmt

# documentation:
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/inputters/text_inputter.py

def model():
    return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",  # see data.yml
          embedding_file_key="word_embeddings",  # The data configuration key of the embedding file.
          embedding_file_with_header=True,  # ``True`` if the embedding file starts with a header line like in GloVe embedding files
          case_insensitive_embeddings=True,
          embedding_size=None),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_file_key="word_embeddings",  # The data configuration key of the embedding file.
          embedding_file_with_header=True,  # ``True`` if the embedding file starts with a header line like in GloVe embedding files
          case_insensitive_embeddings=True,
          embedding_size=None),
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
