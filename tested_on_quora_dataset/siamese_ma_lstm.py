import tensorflow as tf
from keras.layers import  Input, Embedding, LSTM
from keras.layers.merge import concatenate
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda

manhatten = lambda x : K.exp(-K.sum(K.abs(x[0] - x[1]) , axis = 1, keepdims = True))

def conv_dense_block(shape = 10, size = 20, output = 100):
  conv_layers = [[32, 7], [32, 7], [32, 3],[32, 3]]
  input = tf.keras.Input(shape = (None, None, size))
  conv_list = [tf.keras.layers.Conv1D(filter_num, filter_size, activation='relu', padding='causal', input_shape=(None, shape)) for filter_num, filter_size in conv_layers]
  max_pool = tf.keras.layers.Lambda(lambda x : tf.reduce_max(x, axis = 2))
  convolution_output = []
  for conv in conv_list:
      c = conv(input)
      l = max_pool(c)
      convolution_output.append(l)
      x = tf.keras.layers.Concatenate()(convolution_output)
  return tf.keras.Model(inputs=input, outputs = x, name = 'conv_block') 


# NN Model design
def CharSiamese(num_words = 200000, num_chars = 69, EMBEDDING_DIM = 300, MAX_SEQUENCE_LENGTH = 60, MAX_CHAR_SEQUENCE_LENGTH = 10, EMBEDDING_CHAR_DIM = 20, embedding_matrix = None):

    # The embedding layer containing the word vectors
    weights = [embedding_matrix] if embedding_matrix != None else None
    trainable = False if embedding_matrix != None else True 
    emb_layer = Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=weights,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=trainable
    )    

    # The embedding layer containing character vector
    emb_char_layer = Embedding(
        input_dim=num_chars,
        output_dim=EMBEDDING_CHAR_DIM,
        trainable=True
    )

    # LSTM layer
    lstm_layer = LSTM(50)


    # Define inputs
    seq1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    seq2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # char sequence inputs
    char_seq1 = Input(shape=(MAX_SEQUENCE_LENGTH, MAX_CHAR_SEQUENCE_LENGTH,), dtype='int32')
    char_seq2 = Input(shape=(MAX_SEQUENCE_LENGTH, MAX_CHAR_SEQUENCE_LENGTH,), dtype='int32')

    # Run inputs through embedding
    char_emb1 = emb_char_layer(char_seq1)
    char_emb2 = emb_char_layer(char_seq2)

    #CNN, max-pooling and Dense layers
    conv_ = conv_dense_block(size = EMBEDDING_CHAR_DIM)
    conv_emb1 = conv_(char_emb1)
    conv_emb2 = conv_(char_emb2)

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Concat character features
    emb1 = concatenate([emb1 , conv_emb1])
    emb2 = concatenate([emb2 , conv_emb1])

    # Run through LSTM layers
    lstm_a = lstm_layer(emb1)
    lstm_b = lstm_layer(emb2)

    # concat = concatenate([lstm_a, lstmb])

    dist = Lambda(manhatten)([lstm_a, lstm_b])

    return Model(inputs=[seq1, seq2, char_seq1, char_seq2], outputs=dist, name = "baseline")