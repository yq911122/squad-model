import tensorflow as tf
# from util import get_record_parser, get_batch_dataset

class IteratorManager(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.iterator = dataset.make_one_shot_iterator()
        self.handle = tf.placeholder(tf.string, shape=[])
        self.count = 0
        self.string_handle = None

    def may_update_string_handle(self, size, sess):
        self.count += size
        # if self.count >= self.total:
        #     print "count %s is over total %s" % (self.count, self.total)
        #     sess.run(self.iterator.initializer)
        #     self.count = 0
        return self.string_handle

    def get_string_handle(self, sess):
        self.string_handle = sess.run(self.iterator.string_handle())
        return self.string_handle

    def make_feed_dict(self):
        return {
            self.handle: self.string_handle
        }



class LearningRateUpdater(object):

    def __init__(self, patience, init_lr, loss_save=100.0):
        self.lr = init_lr
        self.patience = patience
        self.curr_patience = 0.
        self.loss_save = loss_save

    def update(self, loss, global_step):
        if loss < self.loss_save:
            self.loss_save = loss
            self.curr_patience = 0
        else:
            self.curr_patience += 1
        if self.curr_patience >= self.patience:
            prev_lr = self.lr
            self.lr /= 1.5
            self.loss_save = loss
            self.curr_patience = 0
            print "step: %s-update learning rate from %s to %s" % (global_step, prev_lr, self.lr)
        return self.lr

    def assign(self, sess, model):
        sess.run(model.lr.assign(self.lr))

    def setZero(self, sess, model):
        sess.run(model.lr.assign(0.))

class SummaryWriter(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def write_summaries(self, summ, global_step):
        for s in summ:
            self.writer.add_summary(s, global_step)

    def flush(self):
        self.writer.flush()