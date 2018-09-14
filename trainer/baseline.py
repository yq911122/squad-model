import tensorflow as tf

import model_helper
# import embed
import models
from models import DropoutRNN


class BaseLineModel(object):

    def __init__(self, config, batch, word_mat=None, is_train=True):

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        self.is_train = is_train

        self.inference(config, batch)

        if is_train:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

            summ = []
            for g, v in grads:
                if g is not None:
                    #print(format(v.name))
                    grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                    # sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    summ.append(grad_hist_summary)
                    # summ.append(sparsity_summary)
                    # summ.append(mean_summary)
            self.grad_summ = tf.summary.merge(summ)

        self.saver = tf.train.Saver()


    def inference(self, config, batch):
        
        K, H, N  = config.num_encoding_layers, config.hidden, config.batch_size

        cw, qw, _, _, y1, y2, self.qa_id = batch.get_next()
        q_mask, q_len = model_helper.get_mask_and_len(qw)
        c_mask, c_len = model_helper.get_mask_and_len(cw)
        q_maxlen = qw.get_shape().as_list()[1]
        c_maxlen = cw.get_shape().as_list()[1]

        weight_summ = []
        with tf.variable_scope("embedding"):
            q = tf.nn.embedding_lookup(self.word_mat, qw)
            c = tf.nn.embedding_lookup(self.word_mat, cw)

        with tf.variable_scope("encoding"):
            with tf.variable_scope("context"):
                encoder_cell = tf.nn.rnn_cell.GRUCell(H)
                encoder_outputs, _ = tf.nn.dynamic_rnn(encoder_cell, c, sequence_length=c_len, initial_state=encoder_cell.zero_state(N, dtype=tf.float32))
                weight_summ += model_helper.get_rnn_cell_weight_summ(encoder_cell)

            with tf.variable_scope("question"):
                q_encoder_cell = tf.nn.rnn_cell.GRUCell(H)
                _, encoder_state = tf.nn.dynamic_rnn(q_encoder_cell, q, sequence_length=q_len, initial_state=q_encoder_cell.zero_state(N, dtype=tf.float32))
                weight_summ += model_helper.get_rnn_cell_weight_summ(encoder_cell)

        with tf.variable_scope("decoder"):
            # Create an attention mechanism
            decoder_emb_inp = tf.constant(0., shape=[N, 2, H])

            decoder_lengths = tf.constant(2, shape=[N])
            # projection_layer = tf.layers.Dense(c_maxlen)

            decoder_cell = tf.nn.rnn_cell.GRUCell(H)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(H, encoder_outputs, memory_sequence_length=c_len)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=H)
            initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=N)
            initial_state = initial_state.clone(cell_state=encoder_state)
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
            W = tf.get_variable("W", [H, c_maxlen], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)
            b = tf.get_variable("b", [c_maxlen], initializer=tf.constant_initializer(0.), dtype=tf.float32)

            outputs = tf.reshape(outputs.rnn_output, shape=[2, N, H])

            outputs = [tf.nn.xw_plus_b(o, W, b) for o in tf.unstack(outputs)]

            self.st_logits = model_helper.softmax_mask(outputs[0], c_mask)
            self.end_logits = model_helper.softmax_mask(outputs[1], c_mask)
            
            weight_summ += model_helper.get_rnn_cell_weight_summ(decoder_cell)
            weight_summ.append(tf.summary.histogram("{}/weight".format(W.name), W))
            weight_summ.append(tf.summary.histogram("{}/weight".format(b.name), b))
            self.weight_summ = tf.summary.merge(weight_summ)

        with tf.variable_scope("predict"):
            self.loss, (self.yp1, self.yp2) = models.pointer_boundary_loss_and_prediction(y1, y2, self.st_logits, self.end_logits)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
