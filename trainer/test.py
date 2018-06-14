
import tensorflow as tf
import models.modules as m

def test_rnn_dense():
    inputs = tf.constant(0., shape=[2, 3, 4])
    outputs = m.rnn_dense(inputs, 3)

    expected_output_shape = [2, 3, 3]
    test_shape(outputs, exp_shape=expected_output_shape)

def test_pointer():
    model = m.Pointer(75)
    query = tf.constant(0., shape=[64, 150])
    memory = tf.constant(0., shape=[64, 338, 150])
    mask = tf.constant(1, shape=[64, 338])
    attended_matrix, logits = model(query, mask, memory)
    expected_attended_matrix_shape = [64, 150]
    expected_logits_shape = [64, 338]

    test_shape(attended_matrix, exp_shape=expected_attended_matrix_shape)
    test_shape(logits, exp_shape=expected_logits_shape)

def test_attention():

    model = m.Attention(5)
    query = tf.constant(0., shape=[2, 3, 7])
    memory = tf.constant(0., shape=[2, 4, 8])
    mask = tf.constant(1, shape=[2, 3])
    attended_matrix, logits = model(query, mask, memory)
    expected_attended_matrix_shape = [2, 4, 5]
    expected_logits_shape = [2, 4, 3]

    test_shape(attended_matrix, exp_shape=expected_attended_matrix_shape)
    test_shape(logits, exp_shape=expected_logits_shape)


def test_attention_pool():

    model = m.AttentionPool(5)
    query = tf.constant(0., shape=[2, 3, 7])
    mask = tf.constant(1, shape=[2, 3])
    attended_matrix, logits = model(query, mask)
    expected_attended_matrix_shape = [2, 7]
    expected_logits_shape = [2, 3]

    test_shape(attended_matrix, exp_shape=expected_attended_matrix_shape)
    test_shape(logits, exp_shape=expected_logits_shape)


def test_dropout_rnn():

    model = m.DropoutRNN(2, 5)
    inputs = tf.constant(0., shape=[2, 3, 7])
    outputs = model(inputs)
    expected_outputs_shape = [2, 3, 5]

    test_shape(outputs, exp_shape=expected_outputs_shape)

def test_dropout_birnn():

    model = m.DropoutBiRNN(2, 5)
    inputs = tf.constant(0., shape=[2, 3, 7])
    outputs = model(inputs)
    expected_outputs_shape = [2, 3, 10]

    test_shape(outputs, exp_shape=expected_outputs_shape)


def test_attention_pointer_net():

    model = m.AttentionPointerNet(75)
    init_state = tf.constant(0., shape=[64, 150])
    memory = tf.constant(0., shape=[64, 338, 150])
    mask = tf.constant(1, shape=[64, 338])
    y1, y2 = model(init_state, memory, mask)
    expected_y1_shape = [64, 338]
    expected_y2_shape = [64, 338]

    test_shape(y1, exp_shape=expected_y1_shape)
    test_shape(y2, exp_shape=expected_y2_shape)


def test_attention_bi_rnn():

    model = m.AttentionBiRNN(1, 5)
    query = tf.constant(0., shape=[2, 3, 7])
    memory = tf.constant(0., shape=[2, 4, 8])
    mask = tf.constant(1, shape=[2, 3])
    seqlen = None
    res = model(query, memory, mask, seqlen, True)
    expected_res_shape = [2, 4, 10]

    test_shape(res, exp_shape=expected_res_shape)


def test_gru():
    cell = m._create_cell(150, 1.0, "gru", "gru", is_train=True)
    inputs = tf.constant(0., shape=[64, 150])
    state = tf.constant(0., shape=[64, 150])
    _, res = cell(inputs, state=state)
    exp_shape = [64, 150]

    test_shape(res, exp_shape)


def test_shape(op, exp_shape, feed_dict={}):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(op, feed_dict=feed_dict)
        res_shape = list(result.shape)
        assert exp_shape == res_shape, "expected result shape is inconsistent with the actual shape: %s != %s" % (exp_shape, res_shape)


def main():
    # test_gru()
    test_attention_pointer_net()

if __name__ == '__main__':
    main()
