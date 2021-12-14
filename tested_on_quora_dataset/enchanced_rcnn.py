from keras.layers import Dense, Input, Embedding, Dropout, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.layers import Bidirectional
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Permute

# have not added gating fusion

def cnn_1(size = 60):
  input = Input(shape = (None, size))
  conv1 = tf.keras.layers.Conv1D(64, 1, activation = 'relu')(input)
  conv2 = tf.keras.layers.Conv1D(64, 2, activation = 'relu')(conv1)
  avg =  tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(conv2)
  max_pool = tf.keras.layers.GlobalMaxPooling1D(data_format='channels_first')(conv2)
  o = concatenate([avg, max_pool], axis = 1)
  return Model(inputs = input, outputs = o, name = "cnn_1")

def cnn_2(size = 60):
  input = Input(shape = (None, size))
  conv1 = tf.keras.layers.Conv1D(64, 1 , activation = 'relu')(input)
  avg =  tf.keras.layers.GlobalAveragePooling1D()(conv1)
  max_pool = tf.keras.layers.GlobalMaxPooling1D()(conv1)
  o = concatenate([avg, max_pool])
  return Model(inputs = input, outputs = o, name = "cnn_2")

def cnn_3(size = 60):
  input = Input(shape = (None, size))
  conv1 = tf.keras.layers.Conv1D(64, 1, activation = 'relu')(input)
  conv2 = tf.keras.layers.Conv1D(64, 3, activation = 'relu')(conv1)
  avg =  tf.keras.layers.GlobalAveragePooling1D()(conv2)
  max_pool = tf.keras.layers.GlobalMaxPooling1D()(conv2)
  o = concatenate([avg, max_pool])
  return Model(inputs = input, outputs = o, name = "cnn_3")


def soft_align(size):
  input1, input2 = Input((None,size)) , Input((None,size))
  attention = Lambda(lambda x : tf.linalg.matmul(x[0], x[1], transpose_b=True))([input1, input2])
  w_att_1 = tf.keras.layers.Softmax( axis = 1)(attention)
  w_att_2 = tf.keras.layers.Softmax( axis = 2 )(attention)
  in1_aligned = Lambda(lambda x : tf.linalg.matmul(x[0], x[1]))([w_att_1, input2])
  in2_aligned = Lambda(lambda x : tf.linalg.matmul(x[0], x[1]))([w_att_2, input1])
  return Model(inputs = [input1, input2], outputs = [in1_aligned, in2_aligned])


def mean_max(size):
  input = Input((None, size))
  o1 = tf.keras.layers.GlobalAveragePooling1D()(input)
  o2 = tf.keras.layers.GlobalAveragePooling1D()(input)
  return Model(inputs = input , outputs = [o1, o2])

def enhancedRCNN(num_words = 200000, EMBEDDING_DIM = 300, MAX_SEQUENCE_LENGTH = 60, embedding_matrix = None):

    # inputs
    seq1 = Input(shape = (MAX_SEQUENCE_LENGTH,))
    seq2 = Input(shape = (MAX_SEQUENCE_LENGTH,))

    # Embedding Layer
    weights = [embedding_matrix] if embedding_matrix != None else None
    trainable = False if embedding_matrix != None else True 
    emb_layer = Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=weights,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=trainable
    )

    # Embedding
    emb_1 = emb_layer(seq1)
    emb_2 = emb_layer(seq2)

    # BiGRU layer
    rnn_1 = Bidirectional(GRU(192 ,dropout=0.2, return_sequences = True))(emb_1)
    rnn_2 = Bidirectional(GRU(192 ,dropout=0.2, return_sequences = True))(emb_2)

    # Rotate 
    rnn_1_permuted = Permute((2,1))(rnn_1)
    rnn_2_permuted = Permute((2,1))(rnn_2)

    size = rnn_1_permuted.shape[1]
    cnns = [cnn_1(), cnn_2(), cnn_3()]

    # rcnn encoding
    sentence1_cnn = concatenate([cnns[0](rnn_1_permuted), cnns[1](rnn_1_permuted), cnns[2](rnn_1_permuted)])
    sentence2_cnn = concatenate([cnns[0](rnn_2_permuted), cnns[1](rnn_2_permuted), cnns[2](rnn_2_permuted)])

    #attention
    c1, c2 = soft_align(size)([rnn_1, rnn_2])

    # interactive modelling
    va = tf.subtract(concatenate([rnn_1, c1, rnn_1], axis = 0),  concatenate([rnn_1, tf.multiply(rnn_1, c1)], axis = 0))
    vb = tf.subtract(concatenate([rnn_2, c2, rnn_2], axis = 0),  concatenate([rnn_2, tf.multiply(rnn_2, c2)], axis = 0))

    M = mean_max(size)
    sentence1_att_mean, sentence1_att_max =  M(va)
    sentence2_att_mean, sentence2_att_max = M(vb)


    #merge
    sentence1 = concatenate([sentence1_att_mean, sentence1_cnn, sentence1_att_max])
    sentence2 = concatenate([sentence2_att_mean, sentence2_cnn, sentence2_att_max])

    #subtract
    subtract = Lambda(lambda x : x[0] - x[1])([sentence1, sentence2])
    d1 = Dense(256, activation = 'relu')(subtract)

    #multiply
    subtract = Lambda(lambda x : x[0] * x[1])([sentence1, sentence2])
    d2 = Dense(256, activation = 'relu')(subtract)


    #final layer
    c = concatenate([d1, d2])
    merged = BatchNormalization()(c)
    merged = Dropout(0.2)(merged)

    merged = Dense(133, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    d3 = Dense(1, activation = 'sigmoid')(c)

    return Model(inputs = [seq1, seq2],  outputs = d3, name = "enhancedRCNN")