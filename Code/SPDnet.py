import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Covariance pooling layer
def _cal_cov_pooling(features):
    shape_f = features.get_shape().as_list()
    shape_f = tf.shape(features)
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]), 2)
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    # 0.001 is regularization factor so that the matrix is SPD Matrix
    trace_t = 0.001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)

'''
# Gaussian pooling layer (alternative to covariance pooling)
def _cal_gaussian_pooling(features):
    shape_f = features.get_shape().as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    print('center batch {}'.format(centers_batch.shape))
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch_tile = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch_tile)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    cov = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    cov = tf.add(cov, tf.matmul(tf.transpose(centers_batch,[0,2,1]), centers_batch))
    col_right = tf.reshape(centers_batch, [shape_f[0], shape_f[2], 1])
    new_mat = tf.concat([cov,col_right],2)
    row_bottom = tf.concat([centers_batch,tf.ones([shape_f[0],1,1])],2)
    features_t = tf.concat([new_mat,row_bottom],1)
    shape_f = features_t.get_shape().as_list()
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)
'''

# LogEig Layer
def _cal_log_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.log(s_f)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t

# computes weights for BiMap Layer
def _variable_with_orth_weight_decay(name1, shape, old_shape, new_shape):
    s1 = tf.cast(old_shape, tf.int32)
    s2 = tf.cast(new_shape, tf.int32)
    w0_init, _ = tf.qr(tf.random_normal([s1, s2], mean=0.0, stddev=1.0),full_matrices=False)
    w0 = tf.get_variable(name1, initializer=w0_init)
    tmp1 = tf.expand_dims(w0,0)
    tmp2 = tf.expand_dims(tf.transpose(w0), 0)
    tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
    tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
    return tmp1, tmp2

def _variable_with_orth_weight_decay_decoder(name1, shape, old_shape, new_shape):
    s1 = tf.cast(old_shape, tf.int32)
    s2 = tf.cast(new_shape, tf.int32)
    w0_init, _ = tf.qr(tf.random_normal([s2, s1], mean=0.0, stddev=1.0),full_matrices=False)
    w0 = tf.get_variable(name1, initializer=w0_init)
    tmp1 = tf.expand_dims(tf.transpose(w0),0)
    tmp2 = tf.expand_dims(w0, 0)
    tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
    tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
    return tmp1, tmp2

#
def _cal_rect_cov(features):
    s_f, v_f = tf.self_adjoint_eig(features)
    s_f = tf.clip_by_value(s_f, 1e-4, 10000)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t
