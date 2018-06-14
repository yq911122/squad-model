import tensorflow as tf
import modules as m


def _scaled_dot_similarity(context, query):
    dim = tf.shape(query)[2]
    similarity_matrix = tf.matmul(context, tf.transpose(
        query, perm=[0, 2, 1])) / (dim ** 0.5)
    return similarity_matrix

def _trilinear_similarity(context, query):

    def _element_wise_prod(a, b):
        JA = tf.shape(a)[1]
        JB = tf.shape(b)[1]
        a_ = tf.tile(
            tf.expand_dims(a, [2]), [1, 1, JB, 1])
        b_ = tf.tile(
            tf.expand_dims(b, [2]), [1, 1, JA, 1])
        return tf.multiply(a_, b_)

    dim = tf.shape(query)[2]
    w0 = tf.get_variable("w0", shape=[2*dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    w1 = tf.get_variable("w1", shape=[2*dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    w2 = tf.get_variable("w2", shape=[2*dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    sim_c = tf.matmul(w0, context)
    sim_q = tf.matmul(w1, query)
    sim_q_c = tf.matmul(w2, _element_wise_prod(context, query))
    return tf.add(tf.add(sim_c, sim_q), sim_q_c)

def get_similarity_func(func_name):
    func_map = {
        "scaled_dot": _scaled_dot_similarity,
        "trilinear": _trilinear_similarity
    }

class Attention(object):
    def __init__(self, similarity_func='scaled_dot', hidden=None, keep_prob=1.0, is_train=True, scope="attention"):
        if isinstance(similarity_func, str):
            similarity_func_name = similarity_func
            similarity_func = get_similarity_func(similarity_func_name)
            if similarity_func is None:
                raise Exception("Unkonwn similarity function: %s" % (similarity_func_name))
        self.similarity_func = similarity_func
        self.hidden = hidden
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.scope = scope

    def __call__(self, query, context, mask):
        with tf.variable_scope(self.scope):
            self.query_, self.context_ = query, context
            if self.hidden:
                self.query_ = m.rnn_dense(self.query_, self.hidden, "query")
                self.context_ = m.rnn_dense(self.context_, self.hidden, "context")
            self.similarity_matrix = self.similarity_func(self.context_, self.query_)
            JX = tf.shape(self.similarity_matrix)[1]
            expand_mask = tf.tile(tf.expand_dims(mask, 1), [1, JX, 1])
            self.logits = tf.nn.softmax(softmax_mask(self.similarity_matrix, expand_mask))
            self.attended_vector = tf.matmul(self.logits, self.query_)
            return self.attended_vector