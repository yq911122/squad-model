import tensorflow as tf

import modules as m
from embed import TrainableEmbedding
import attention as a

class QANet(object):
    """"""

    def __init__(self, config, batch, word_mat=None, char_mat=None, is_train=True):

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.inference(config, batch)

        if is_train:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)     

    def char_embedding(self, x, emb_mat):
        embed = tf.nn.embedding_lookup(emb_mat, x)
        return tf.reduce_max(embed, axis=2)

    def context_query_attention_layer(self, c, q):
        attention = a.Attention(similarity_func='trilinear')
        A = attention(c, q)
        sim = attention.similarity_matrix
        sim_norm = tf.nn.softmax(sim, axis=1)
        B = tf.matmul(tf.matmul(attention.logits, sim_norm), c)
        return A, B

    def model_encoding_layer(self, c, A, B, config):
        inputs = tf.concat([c, A, B])

        params = config.num_conv_layers_model, config.kernel_size, config.embed_size, config.num_conv_filters, \
                config.conv_out_channels, config.attention_heads, config.attention_out_dim, config.feedforward_out_dim

        outputs = [inputs]
        
        for _ in xrange(config.num_model_blocks):
            encoder_block = m.EncoderBlock(*params)
            output = encoder_block(outputs[-1], c_mask, c_maxlen)

        return outputs[1:]


    def inference(self, config, batch):

        K, H, N, CH, CL = config.num_encoding_layers, config.hidden, config.batch_size, config.char_hidden, config.char_limit

        c, q, ch, qh, y1, y2, qa_id = batch.get_next()
        q_mask, q_len = m.get_mask_and_len(q)
        c_mask, c_len = m.get_mask_and_len(c)

        c_maxlen = tf.reduce_max(c_len)
        q_maxlen = tf.reduce_max(q_len)

        with tf.variable_scope("embedding"):
            with tf.variable_scope("word"):
                emb = TrainableEmbedding(self.word_mat)
                qw_emb = emb.get_embedding(q)
                cw_emb = emb.get_embedding(c)

            with tf.variable_scope("char"):
                qc_emb = self.char_embedding(qh, self.char_mat)
                cc_emb = self.char_embedding(ch, self.char_mat)
            q_emb = tf.concat([qw_emb, qc_emb], axis=2)
            c_emb = tf.concat([cw_emb, cc_emb], axis=2)

        q, c = q_emb, c_emb

        with tf.variable_scope("embed_encoding"):
            params = config.num_conv_layers, config.kernel_size, config.embed_size, config.num_conv_filters, \
                config.conv_out_channels, config.attention_heads, config.attention_out_dim, config.feedforward_out_dim
            with tf.variable_scope('c'):
                encoder_block = m.EncoderBlock(*params)
                c = encoder_block(c, c_mask, c_maxlen)
            with tf.variable_scope('q'):
                encoder_block = m.EncoderBlock(*params)
                q = encoder_block(q, q_mask, q_maxlen)

        with tf.variable_scope("context_query_attention"):
            A, B = self.context_query_attention_layer(c, q)

        with tf.variable_scope("model_encoding"):
            M0, M1, M2 = self.model_encoding_layer(c, A, B, config)

        with tf.variable_scope("predict"):
            W0 = tf.get_variable("W0", shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
            W1 = tf.get_variable("W1", shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.st_logits = tf.nn.softmax(tf.matmul(W0, tf.concat([M0, M1], axis=2)))
            self.end_logits = tf.nn.softmax(tf.matmul(W1, tf.concat([M0, M2], axis=2)))
            self.loss, (self.yp1, self.yp2) = m.pointer_boundary_loss_and_prediction(y1, y2, self.st_logits, self.end_logits)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step



