import tensorflow as tf

def attn(size = 200, k = 64, v = 64, name = "attn", decay_m = None):
  input = tf.keras.layers.Input(shape = (None, size))
  q_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(k, use_bias=False))
  k_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(k, use_bias=False))
  v_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(v, use_bias=False))
  softmax = tf.keras.layers.Softmax()

  query_vec = q_dense(input)
  k_vec = k_dense(input)
  v_vec = v_dense(input)
  d_k = tf.constant(v_vec.shape[-1], dtype=tf.float32)
  if decay_m is not None:
    attn = softmax(tf.matmul(query_vec , k_vec , transpose_b=True) / tf.sqrt(d_k)) + 0.01 * decay_m
  else:
    attn = softmax(tf.matmul(query_vec , k_vec , transpose_b=True) / tf.sqrt(d_k))
  attn = tf.matmul(attn, v_vec)
  return tf.keras.Model(inputs = input, outputs = attn, name = name)

def multi_attn(size = 200, k = 64, v = 64, out_dim = 64, h = 4, decay_m = None, name = "multi_attn"):
  input = tf.keras.layers.Input(shape = (None, size))
  attns = [attn(name = f"attn_{i}", decay_m = decay_m, size = size)(input) for i in range(h)]
  multi_attns = tf.keras.layers.concatenate(attns)
  linear = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dim, use_bias=False))(multi_attns)
  return tf.keras.Model(inputs = input, outputs = linear, name = name)

def cross_attn(size = 64, k = 64, v = 64, name = "cross_attn"):
  input1, input2= tf.keras.layers.Input(shape = (None, size)), tf.keras.layers.Input(shape = (None, size))
  q_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(k, use_bias=False))
  k_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(k, use_bias=False))
  v_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(v, use_bias=False))
  softmax = tf.keras.layers.Softmax()

  query_vec = q_dense(input1)
  k_vec = k_dense(input2)
  v_vec = v_dense(input2)
  d_k = tf.constant(v_vec.shape[-1], dtype=tf.float32)
  attn = softmax(tf.matmul(query_vec , k_vec , transpose_b=True) / tf.sqrt(d_k))
  attn = tf.matmul(attn, v_vec)
  return tf.keras.Model(inputs = [input1, input2], outputs = attn, name = name)


def multi_cross_attn(size = 64, k = 64, v = 64, out_dim = 64, h = 4):
  input1 , input2 = tf.keras.layers.Input(shape = (None, size)), tf.keras.layers.Input(shape = (None, size))
  attns = [cross_attn(name = f"cross_attn_{i}", size = size)([input1, input2]) for i in range(h)]
  multi_attns = tf.keras.layers.concatenate(attns)
  linear = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dim, use_bias=False))(multi_attns)
  return tf.keras.Model(inputs = [input1, input2], outputs = linear, name = "multi_cross_attn")


def conv_block(size = 128, a = 256, b = 562, c = 256, name = "conv_block"):
  input = tf.keras.layers.Input(shape = (None, size))
  c1 = tf.keras.layers.Conv1D(a, 1, padding = "same")
  c2 = tf.keras.layers.Conv1D(b, 2, padding = "same")
  c3 = tf.keras.layers.Conv1D(c, 3, padding = "same")
  o = tf.keras.layers.concatenate([c1(input), c2(input), c3(input)], axis = 2)
  return tf.keras.Model(inputs = input, outputs = o, name = name)

# model : DARCNN

def darcnn(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM):
    # constant decay by dist
    M = tf.constant([[-abs(i-j) for j in range(MAX_SEQUENCE_LENGTH)] for i in range(MAX_SEQUENCE_LENGTH)],  dtype = tf.float32, name="decay_const")

    # input
    seq1 , seq2 = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,)), tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Embedding
    emb = tf.keras.layers.Embedding(
        input_dim = num_words,
        output_dim = EMBEDDING_DIM,
        weights = [embedding_matrix],
        trainable = False
    )

    emb1, emb2 = emb(seq1) , emb(seq2)

    # Bilstm
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True, dropout=0.2, name = "lstm"))
    rnn1 = bilstm(emb1)
    rnn2 = bilstm(emb2)

    # self - attention
    self_attn = multi_attn(decay_m = None, size = rnn1.shape[-1], out_dim = rnn1.shape[-1])

    attn1 = self_attn(rnn1)
    attn2 = self_attn(rnn2)


    # decay - attention
    self_decay_attn1 = multi_attn(decay_m = M, name = "decay_multi_attn", size = rnn1.shape[-1], out_dim = rnn1.shape[-1])

    decay_attn1 = self_decay_attn1(rnn1)
    decay_attn2 = self_decay_attn1(rnn2)

    # cross - attention
    c_attn = multi_cross_attn(size = decay_attn1.shape[-1], out_dim = decay_attn1.shape[-1])

    cr_1 = c_attn([decay_attn1, decay_attn2])
    cr_2 = c_attn([attn1, attn2])
    cr_1 = tf.keras.layers.concatenate([cr_1, decay_attn1])
    cr_2 = tf.keras.layers.concatenate([cr_2, attn1])

    cr_4 = c_attn([attn2, attn1])
    cr_3 = c_attn([decay_attn2, decay_attn1])
    cr_4 = tf.keras.layers.concatenate([cr_4, attn2])
    cr_3 = tf.keras.layers.concatenate([cr_3, decay_attn2])


    # Add and normalize
    normalize = tf.keras.layers.LayerNormalization()
    s1 = normalize(tf.add(cr_1, cr_2))
    s2 = normalize(tf.add(cr_3, cr_4))

    # cnn block
    c1 = conv_block(name = "conv1", size = s1.shape[-1])
    o1 = c1(s1)
    o2 = c1(s2)

    c2 = conv_block(name = "conv2", size = o1.shape[-1])
    o1 = c2(o1) 
    o2 = c2(o2)

    # 1 max pooling
    glb_max_pool = tf.keras.layers.GlobalMaxPool1D()
    o1 = glb_max_pool(o1)
    o2 = glb_max_pool(o2)
    z = tf.keras.layers.concatenate([o1, o2])

    # 2 MLP layers
    z = tf.keras.layers.Dense(1024, activation='relu')(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Dropout(0.5)(z)

    # sigmoid
    z = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs = [seq1, seq2], outputs = z, name = "DARCNN")
    return model