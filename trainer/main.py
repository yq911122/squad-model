import tensorflow as tf
import numpy as np
import os

from rnet import RNET
from match_lstm import MatchLSTM
from util import get_record_parser, evaluate, get_batch_dataset, load_from_gs, write_to_gs

def get_model(model_type):
    if model_type == "rnet":
        return RNET
    if model_type == "match_lstm":
        return MatchLSTM
    raise Exception("Not supported model type %s" % (model_type))

def train(config):
    word_mat = np.array(load_from_gs(config.word_emb_file), dtype=np.float32)
    char_mat = np.array(load_from_gs(config.char_emb_file), dtype=np.float32)
    train_eval_file = load_from_gs(config.train_eval_file)
    dev_eval_file = load_from_gs(config.dev_eval_file)
    meta = load_from_gs(config.dev_meta)

    dev_total = meta["total"]

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_text_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_text_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model_cls = get_model(config.model)
    model = model_cls(config, iterator, word_mat, char_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in xrange(1, config.num_steps + 1):
            global_step = sess.run(model.global_step) + 1

            # v, h, state = sess.run([model.v, model.h, model.init_state], feed_dict={
            #                           handle: train_handle})
            # print("======")
            # print(v.shape)
            # print(h.shape)
            # print(state.shape)
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                      handle: train_handle})
            if global_step % config.period == 0:
                tf.logging.info("training step: step {} adding summary".format(global_step))
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            if global_step % config.checkpoint == 0:
                tf.logging.info("training step: step {} checking the model".format(global_step))
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                _, summ = evaluate_batch(
                    model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)

                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in xrange(1, num_batches + 1):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def test(config):
    word_mat = np.array(load_from_gs(config.word_emb_file), dtype=np.float32)
    char_mat = np.array(load_from_gs(config.char_emb_file), dtype=np.float32)
    eval_file = load_from_gs(config.test_eval_file)
    meta = load_from_gs(config.test_meta)

    total = meta["total"]

    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model_cls = get_model(config.model)
    model = model_cls(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in xrange(total // config.batch_size + 1):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        write_to_gs(config.answer_file, remapped_dict)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))
