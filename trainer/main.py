import tensorflow as tf
import numpy as np
import os

from match_lstm import MatchLSTM
from baseline import BaseLineModel
from util import get_record_parser, evaluate, get_batch_dataset, load_from_gs, write_to_gs, convert_tokens
import util
from train_helper import *

def get_model(model_type):
    if model_type == "rnet":
        return RNET
    if model_type == "match_lstm":
        return MatchLSTM
    raise Exception("Not supported model type %s" % (model_type))


def train(config):
    word_mat = np.array(load_from_gs(config.word_emb_file), dtype=np.float32)
    train_eval = load_from_gs(config.train_eval_file)
    dev_eval = load_from_gs(config.dev_eval_file)

    print("Building model...")
    # data_manager = DataManager(config)

    train_graph = tf.Graph()
    dev_graph = tf.Graph()

    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)

    # initialize train model and dev model separately
    with train_graph.as_default():
        train_iterator_manager = IteratorManager(train_dataset)
        train_model = BaseLineModel(config, train_iterator_manager.iterator, word_mat)
        initializer = tf.global_variables_initializer()

    with dev_graph.as_default():
        dev_iterator_manager = IteratorManager(dev_dataset)
        dev_model = BaseLineModel(config, dev_iterator_manager.iterator, word_mat, is_train=False)

    checkpoints_path = os.path.join(config.save_dir, "checkpoints")

    # initialize train and dev session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    train_sess = tf.Session(graph=train_graph, config=sess_config)
    dev_sess = tf.Session(graph=dev_graph, config=sess_config)

    train_sess.run(initializer)
    train_iterator_manager.get_string_handle(train_sess)
    dev_iterator_manager.get_string_handle(dev_sess)

    summary_writer = SummaryWriter(config.log_dir)

    lr_updater = LearningRateUpdater(patience=config.patience, init_lr=config.init_lr, loss_save=100.0)
    lr_updater.assign(train_sess, train_model)

    # checkpoint_path = tf.train.latest_checkpoint(config.save_dir, latest_filename=None)
    # train_model.saver.restore(train_sess, checkpoint_path)
    
    for _ in xrange(1, config.num_steps + 1):

        global_step = train_sess.run(train_model.global_step) + 1

        loss, train_op, grad_summ, weight_summ = train_sess.run([train_model.loss, train_model.train_op, train_model.grad_summ, train_model.weight_summ], feed_dict=train_iterator_manager.make_feed_dict())
        # tf.logging.info("training step: step {} adding loss: {}".format(global_step, loss))

        if global_step % config.period == 0:
            tf.logging.info("training step: step {} adding loss: {}".format(global_step, loss))
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
            summary_writer.write_summaries([loss_sum], global_step)
            summary_writer.write_summaries([grad_summ], global_step)
            summary_writer.write_summaries([weight_summ], global_step)
            
            lr_summ = tf.Summary(value=[tf.Summary.Value(tag="model/lr", simple_value=lr_updater.lr), ])
            summary_writer.write_summaries([lr_summ], global_step)

            # summary_writer.flush()
                        
            # lr_updater.update(loss)
            # lr_updater.assign(train_sess, train_model)

        if global_step % config.checkpoint == 0:
            lr_updater.setZero(train_sess, train_model)
            tf.logging.info("training step: step {} checking the model".format(global_step))
            checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=global_step)

            _, summ = evaluate_batch(train_model, config.val_num_batches, train_eval, train_sess, "train", train_iterator_manager)
            summary_writer.write_summaries(summ, global_step)
            
            dev_model.saver.restore(dev_sess, checkpoint_path)
            metrics, summ = evaluate_batch(dev_model, config.val_num_batches, dev_eval, dev_sess, "dev", dev_iterator_manager)
            summary_writer.write_summaries(summ, global_step)
            
            summary_writer.flush()
                        
            lr_updater.update(metrics["loss"], global_step)
            lr_updater.assign(train_sess, train_model)

def evaluate_batch(model, num_batches, eval_file, sess, data_type, iterator_manager):
    answer_dict = {}
    losses = []
    for _ in xrange(1, num_batches + 1):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict=iterator_manager.make_feed_dict())
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
