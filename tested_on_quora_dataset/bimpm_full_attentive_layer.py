import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Dropout, LSTM
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.layers import TimeDistributed, Bidirectional
def full_matching(metric, vec, w, num_perspective):
    w = tf.expand_dims(tf.expand_dims(w, 0), 2)  
    metric = w * tf.stack([metric] * num_perspective, axis=1)  
    vec = w * tf.stack([vec] * num_perspective, axis=1)  
    m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))   
    n = tf.norm(metric, axis=3, keepdims=True) * tf.norm(vec, axis=3, keepdims=True)  
    cosine = tf.transpose(tf.math.divide(m, n), [0, 2, 3, 1])   
    return cosine

def cosine_matrix(v1 , v2):
  m = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))
  n = tf.norm(v1, axis=2, keepdims=True) * tf.norm(v2, axis=2, keepdims=True)
  cosine = tf.math.divide(m, n)
  return cosine


def attn_matching(metric, vec, w, num_perspective):
    w = tf.expand_dims(tf.expand_dims(w, 0), 2)  # [1, L, 1, H]
    metric = w * tf.stack([metric] * num_perspective, axis=1)  # [1, L, 1, H] * [B, L, S, H] = [B, L, S, H]
    vec = w * tf.stack([vec] * num_perspective, axis=1)  # [1, L, 1, H] * [B, L, 1, H] = [B, L, 1, H]
    m = tf.reduce_sum(tf.multiply(metric, vec), axis=3, keepdims=True)
    n = tf.norm(metric, axis=3, keepdims=True) * tf.norm(vec, axis=3, keepdims=True)  # [B, L, S, 1]
    cosine = tf.transpose(tf.math.divide_no_nan(m, n), [0, 2, 3, 1])   # [B, S, 1, L]
    return cosine



def full_attentive_matching_bimpm(num_words = 200000, EMBEDDING_DIM = 300, MAX_SEQUENCE_LENGTH = 60, MAX_CHAR_SEQUENCE_LENGTH = 10, embedding_matrix = None):
    # inputs
    seq1 = Input(shape = (MAX_SEQUENCE_LENGTH,))
    seq2 = Input(shape = (MAX_SEQUENCE_LENGTH,))
    char_seq1 = Input(shape=(MAX_SEQUENCE_LENGTH, MAX_CHAR_SEQUENCE_LENGTH,), dtype='int32')
    char_seq2 = Input(shape=(MAX_SEQUENCE_LENGTH, MAX_CHAR_SEQUENCE_LENGTH,), dtype='int32')

    
    # Embedding Layer
    weights = [embedding_matrix] if embedding_matrix != None else None
    emb_layer = Embedding(
      input_dim=num_words,
      output_dim=EMBEDDING_DIM,
      weights=weights,
      input_length=MAX_SEQUENCE_LENGTH,
      trainable=False
    )

    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    #char Embedding Layer
    char_emb_layer = Embedding(
      input_dim=69,
      output_dim=20,
      input_length=MAX_SEQUENCE_LENGTH
    )

    char_lstm = LSTM(50, dropout=0.2)

    # input_1 embedding
    char_emb1 = char_emb_layer(char_seq1)
    char_emb1 = TimeDistributed(char_lstm)(char_emb1)
    emb1 = concatenate([emb1, char_emb1])

    #input_2 embedding
    char_emb2 = char_emb_layer(char_seq2)
    char_emb2 = TimeDistributed(char_lstm)(char_emb2)
    emb2 = concatenate([emb2, char_emb2])


    # Contextual LSTMS
    a_fw, a_bw = Bidirectional(LSTM(100 ,dropout=0.2, return_sequences = True),merge_mode=None)(emb1)
    b_fw, b_bw = Bidirectional(LSTM(100 ,dropout=0.2, return_sequences = True), merge_mode=None)(emb2)

    # Full-Matching Layer
    # for q1
    w1 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w1")
    a_fw_next = full_matching(a_fw, tf.expand_dims(b_fw[:, -1, :], 1), w1, 20)
    w2 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w2")
    a_bw_next = full_matching(a_bw, tf.expand_dims(b_bw[:, -1, :], 1), w1, 20)
    # for q2
    w3 = tf.Variable(tf.random.normal([20 , 100],), trainable=True, name = "w3")
    b_fw_next = full_matching(b_fw, tf.expand_dims(a_fw[:, -1, :], 1), w3, 20)
    w4 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w4")
    b_bw_next = full_matching(b_bw, tf.expand_dims(a_bw[:, -1, :], 1), w4, 20)

    # Attentive Matching Layer
    cos_fw = cosine_matrix(a_fw, b_fw)
    cos_bw = cosine_matrix(a_bw, b_bw)
    #Attn vectors
    a_attn_fw = tf.matmul(cos_fw, a_fw)
    a_attn_bw = tf.matmul(cos_bw, a_bw)
    b_attn_fw = tf.matmul(tf.transpose(cos_fw, [0,2,1]), b_fw)
    b_attn_bw = tf.matmul(tf.transpose(cos_bw, [0,2,1]), b_bw)
    a_mean_fw = tf.math.divide_no_nan(a_attn_fw, tf.reduce_sum(cos_fw, axis=2, keepdims=True))
    a_mean_bw = tf.math.divide_no_nan(a_attn_bw, tf.reduce_sum(cos_bw, axis=2, keepdims=True))
    b_mean_fw = tf.math.divide_no_nan(b_attn_fw, tf.reduce_sum(tf.transpose(cos_fw, [0,2,1]), axis=2, keepdims=True))
    b_mean_bw = tf.math.divide_no_nan(b_attn_bw, tf.reduce_sum(tf.transpose(cos_bw, [0,2,1]), axis=2, keepdims=True))
    # for q1
    w5 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w5")
    a_fw_attn_next = attn_matching(a_fw, a_mean_fw, w5, 20)
    w6 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w6")
    a_bw_attn_next = attn_matching(a_bw, a_mean_bw, w6, 20)
    # for q2
    w7 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w7")
    b_fw_attn_next = attn_matching(b_fw, b_mean_fw, w7, 20)
    w8 = tf.Variable(tf.random.normal([20 , 100]), trainable=True, name = "w8")
    b_bw_attn_next = attn_matching(b_bw, b_mean_bw, w8, 20)


    mv = concatenate([a_fw_next, a_bw_next, a_fw_attn_next, a_bw_attn_next], axis = 2)
    mv = tf.nn.dropout(tf.reshape(mv, (-1, mv.shape[1], mv.shape[2] * mv.shape[3])) , 0.2)

    mb = concatenate([b_fw_next, b_bw_next, b_fw_attn_next, b_bw_attn_next], axis = 2)
    mb = tf.nn.dropout(tf.reshape(mb, (-1, mb.shape[1], mb.shape[2] * mb.shape[3])), 0.2)

    # aggregation layer
    aggr_lstm_layer = Bidirectional(LSTM(20 * 8,dropout=0.2, return_sequences = True), merge_mode=None)
    m_f, m_b = aggr_lstm_layer(mv)
    b_f, b_b = aggr_lstm_layer(mb)

    # Linear layers
    r = concatenate([m_f[:, -1, :], m_b[:, 0 , :] , b_f[:, 0, :], b_b[:, -1, :]])
    merged = Dense(200, activation = 'relu')(r)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(133, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    output = Dense(1, activation = 'sigmoid')(merged)
    return Model(inputs = [seq1, seq2, char_seq1, char_seq2], outputs = output, name ="BMPMFL_ATTN")