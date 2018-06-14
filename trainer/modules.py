
import tensorflow as tf

INF = 1e30

def _create_cell(num_units, output_keep_prob, cell_type, name, reuse=False, is_train=True):
    if cell_type == "gru":
        func = tf.contrib.rnn.GRUCell
    elif cell_type == "lstm":
        func = tf.contrib.rnn.LSTMCell
    else:
        raise Exception("Not supported cell type %s" % (cell_type))
        return 

    cell = func(num_units, name=name, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob if is_train else 1.0)
    return cell

def _build_dynamic_rnn(cell, inputs, seqlen=None, reverse_inputs=False, init_state=None, scope="dynamic_rnn"):
    if reverse_inputs and seqlen is not None:
        inputs = tf.reverse_sequence(inputs, seq_lengths=seqlen, seq_dim=1, batch_dim=0)

    if init_state is not None:
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, 
                            sequence_length=seqlen,
                            initial_state=init_state,
                            scope=scope)
    else:
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, 
                            sequence_length=seqlen,
                            dtype=tf.float32,
                            scope=scope)
    if reverse_inputs and seqlen is not None:
        outputs = tf.reverse_sequence(outputs, seq_lengths=seqlen, seq_dim=1, batch_dim=0)
    return outputs

class BaseClass(object):

    def __init__(self, hidden, dropout_keep_prob=1.0, scope="base_class", is_train=True):
        self.dropout_keep_prob = dropout_keep_prob
        self.scope = scope
        self.hidden = hidden
        self.is_train = is_train


class DropoutRNN(BaseClass):

    def __init__(self, num_layers, hidden, cell_type="gru", dropout_keep_prob=0.5, is_train=True, scope="dropout_rnn"):

        super(DropoutRNN, self).__init__(hidden, dropout_keep_prob, scope, is_train)
        self.cells = []

        with tf.variable_scope(scope):
            for layer in range(num_layers):
                self.cells.append(_create_cell(hidden, dropout_keep_prob, cell_type, "cell_{}".format(layer), is_train=is_train))

    def __call__(self, inputs, seqlen=None, init_state=None, reverse_inputs=False):

        with tf.variable_scope(self.scope):

            if self.dropout_keep_prob:
                inputs = dropout(inputs, self.dropout_keep_prob, self.is_train)
            
            outputs = [inputs]
            for layer, cell in enumerate(self.cells):
                output = _build_dynamic_rnn(cell, outputs[-1], seqlen=seqlen, 
                                reverse_inputs=reverse_inputs, init_state=init_state, 
                                scope="dropout_rnn_layer_{}".format(layer))
                outputs.append(output)

        return outputs[-1]

class DropoutBiRNN(BaseClass):

    def __init__(self, num_layers, hidden, cell_type="gru", dropout_keep_prob=0.5, is_train=True, scope="dropout_bi_rnn"):

        super(DropoutBiRNN, self).__init__(hidden, dropout_keep_prob, scope, is_train)

        self.bw_cells, self.fw_cells = [], []

        with tf.variable_scope(scope):
            for layer in range(num_layers):
                self.fw_cells.append(_create_cell(hidden, dropout_keep_prob, cell_type, "fw_{}".format(layer), is_train=is_train))
                self.bw_cells.append(_create_cell(hidden, dropout_keep_prob, cell_type, "bw_{}".format(layer), is_train=is_train))

    def __call__(self, inputs, seqlen=None, init_state=None):

        with tf.variable_scope(self.scope):

            if self.dropout_keep_prob:
                inputs = dropout(inputs, self.dropout_keep_prob, self.is_train)
            
            outputs = [inputs]
            for layer, (bw_cell, fw_cell) in enumerate(zip(self.bw_cells, self.fw_cells)):
                bw_outputs = _build_dynamic_rnn(bw_cell, outputs[-1], seqlen=seqlen, 
                                reverse_inputs=True, init_state=init_state, 
                                scope="dropout_bi_rnn_layer_{}_fw".format(layer))

                fw_outputs = _build_dynamic_rnn(fw_cell, outputs[-1], seqlen=seqlen, 
                                reverse_inputs=False, init_state=init_state, 
                                scope="dropout_bi_rnn_layer_{}_bw".format(layer))
                layer_outputs = tf.concat([fw_outputs, bw_outputs], axis=2, name="dropout_bi_rnn_layer_{}_output".format(layer))
                outputs.append(layer_outputs)

        return outputs[-1]


class AttentionBiRNN(BaseClass):

    def __init__(self, num_layers, hidden, cell_type="gru", dropout_keep_prob=0.5, is_train=True, scope="attention_bi_rnn"):

        super(AttentionBiRNN, self).__init__(hidden, dropout_keep_prob, scope, is_train)
        self.bi_rnn = DropoutBiRNN(num_layers, hidden, cell_type, dropout_keep_prob, is_train)
        self.attention = Attention(hidden, dropout_keep_prob, is_train)

    def _prep_inputs(self, query, memory, mask, gate):

        with tf.variable_scope(self.scope):

            attended_matrix, _ = self.attention(query, mask, memory)
            res = tf.concat([self.attention.memory, attended_matrix], axis=2)

            if gate:
                with tf.variable_scope("gate"):
                    dim = res.get_shape().as_list()[-1]
                    gate = tf.sigmoid(
                        rnn_dense(res, dim, scope="inputs")
                        )
                res = gate * res
            return res

    def __call__(self, query, memory, mask, seqlen, gate=False):

        inputs = self._prep_inputs(query, memory, mask, gate)
        return self.bi_rnn(inputs, seqlen)


class AttentionPointerNet(BaseClass):

    def __init__(self, hidden, cell_type="gru", dropout_keep_prob=0.5, is_train=True, scope="attention_pointer_net"):
        super(AttentionPointerNet, self).__init__(hidden, dropout_keep_prob, scope, is_train)
        self.cell = _create_cell(hidden * 2, dropout_keep_prob, cell_type, "gru", is_train=is_train)
        self.pointer = Pointer(hidden, dropout_keep_prob, is_train)

    def __call__(self, init_state, memory, mask):

        with tf.variable_scope(self.scope):
            attended_matrix, st_logits = self.pointer(init_state, mask, memory)

            _, state = self.cell(attended_matrix, state=init_state)

            tf.get_variable_scope().reuse_variables()
            _, end_logits = self.pointer(state, mask, memory)
            return st_logits, end_logits


class AttentionLayer(BaseClass):

    def __init__(self, hidden, dropout_keep_prob=1.0, is_train=True, scope="attention"):
        super(AttentionLayer, self).__init__(hidden, dropout_keep_prob, scope, is_train)

    def calculate_similariy_matrix(self, query, memory):
        pass

    def get_attention_from_similariy_matrix(self, similarity_matrix, query, mask):
        JX = tf.shape(similarity_matrix)[1]
        expand_mask = tf.tile(tf.expand_dims(mask, 1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(similarity_matrix, expand_mask))
        attended_matrix = tf.matmul(logits, query)
        return attended_matrix, logits

    def __call__(self, query, mask, memory=None):
        self.query = dropout(query, self.dropout_keep_prob, self.is_train)
        if memory is not None: 
            self.memory = dropout(memory, self.dropout_keep_prob, self.is_train)
        else:
            self.memory = None
        similarity_matrix = self.calculate_similariy_matrix(self.query, self.memory)
        return self.get_attention_from_similariy_matrix(similarity_matrix, self.query, mask)


class AttentionPool(AttentionLayer):
    def calculate_similariy_matrix(self, query, memory=None):
        with tf.variable_scope(self.scope):
            similarity_matrix = rnn_dense(
                tf.tanh(
                    rnn_dense(query, self.hidden, scope="query")
                    ),
                1, scope="logits")
            return tf.squeeze(similarity_matrix, [2])

    def get_attention_from_similariy_matrix(self, similarity_matrix, query, mask):
        logits = tf.nn.softmax(softmax_mask(similarity_matrix, mask))
        expand_logits = tf.expand_dims(logits, axis=2)
        attended_matrix = tf.reduce_sum(expand_logits*query, axis=1)
        return attended_matrix, logits


class Attention(AttentionLayer):
    def calculate_similariy_matrix(self, query, memory):
        with tf.variable_scope(self.scope):
            similarity_matrix = tf.matmul(memory, tf.transpose(
                query, perm=[0, 2, 1])) / (self.hidden ** 0.5)
            return similarity_matrix

    def __call__(self, query, mask, memory=None):
        query_ = tf.nn.relu(
                rnn_dense(query, self.hidden, scope="query")
                )
        memory_ = tf.nn.relu(
            rnn_dense(memory, self.hidden, scope="memory")
            )
        return super(Attention, self).__call__(query_, mask, memory_)


class Pointer(AttentionLayer):

    def calculate_similariy_matrix(self, query, memory):  # query -> states, ie, answer, memory -> context
        with tf.variable_scope(self.scope):
            JX = tf.shape(memory)[1]
            query_ = tf.expand_dims(query, axis=1)
            query_ = tf.tile(query_, [1, JX, 1])
            similarity_matrix = rnn_dense(
                tf.tanh(
                    rnn_dense(tf.concat([memory, query_], 2), self.hidden, scope="query")
                    ),
                1, scope="logits")
            return tf.squeeze(similarity_matrix, [2])

    def get_attention_from_similariy_matrix(self, similarity_matrix, query, mask):
        logits = tf.nn.softmax(softmax_mask(similarity_matrix, mask))
        expand_logits = tf.expand_dims(logits, axis=2)
        attended_matrix = tf.reduce_sum(expand_logits*self.memory, axis=1)
        return attended_matrix, logits


def dropout(inputs, keep_prob, is_train=True):
    return tf.cond(tf.cast(is_train, tf.bool), lambda: tf.nn.dropout(inputs, keep_prob),
        lambda: inputs)



def softmax_mask(val, mask=None):
    if mask is None: return val
    return -INF * (1 - tf.cast(tf.cast(mask, tf.int32), tf.float32)) + val



def _get_shape(inputs, output_dim):
    """get output shape of inputs tf.shape[input][:-1] + [output_dim]. This is generally useful in rnn calculation, where we need to do X * W, X of shape [n_batch, seqlen, num_units], W of shape [num_units, output_dim], outputs of shape [n_batch, seqlen, output_dim]"""
    input_shapes = tf.shape(inputs)
    input_num_dim = inputs.get_shape().as_list()
    weitght_shapes = [input_num_dim[-1], output_dim]
    output_shapes = [input_shapes[idx] for idx in range(len(input_num_dim)-1)] + [output_dim]
    return input_shapes, weitght_shapes, output_shapes



def rnn_dense(inputs, output_dim, scope="rnn_dense", use_bias=False):
    
    with tf.variable_scope(scope):
        input_shapes, weitght_shapes, output_shapes = _get_shape(inputs, output_dim)
        W = tf.get_variable("W", weitght_shapes)
        res = tf.matmul(
            tf.reshape(inputs, [-1, weitght_shapes[0]]), W
            )
        if use_bias:
            b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        outputs = tf.reshape(res, output_shapes, name="outputs")
    return outputs



def get_mask_and_len(v):
    mask = tf.cast(v, tf.bool)
    seqlen = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    return mask, seqlen


def pointer_boundary_loss_and_prediction(y1, y2, logit_y1, logit_y2):
    outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logit_y1), axis=2),
                      tf.expand_dims(tf.nn.softmax(logit_y2), axis=1))
    outer = tf.matrix_band_part(outer, 0, 15)
    yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logit_y1, labels=tf.stop_gradient(y1))
    losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logit_y2, labels=tf.stop_gradient(y2))
    loss = tf.reduce_mean(losses + losses2)
    return loss, (yp1, yp2)






