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


class DataManager(object):

    def __init__(self, config):
        parser = get_record_parser(config)
        self.train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        self.dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        # self.meta = load_from_gs(config.)
        self.batch_size = config.batch_size

        self.iterator_managers = {
            "train": IteratorManager(self.train_dataset),
            "dev": IteratorManager(self.dev_dataset)
            }

    def setup_train_or_dev_string_handles(self, sess, ite_type):
        self.iterator_managers[ite_type].get_string_handle(sess)

    def make_feed_dict(self, data_type):
        return {
            self.handle: self.iterator_managers[data_type].string_handle
        }

    def step(self, sess, data_type):
        self.iterator_managers[data_type].may_update_string_handle(self.batch_size, sess)

class LearningRateUpdater(object):

    def __init__(self, patience, init_lr, loss_save=100.0):
        self.lr = init_lr
        self.patience = patience
        self.curr_patience = 0.
        self.loss_save = loss_save

    def update(self, loss):
        if loss < self.loss_save:
            self.loss_save = loss
            self.curr_patience = 0
        else:
            self.curr_patience += 1
        if self.curr_patience >= self.patience:
            self.lr /= 2.0
            self.loss_save = loss
            self.curr_patience = 0
        return self.lr

    def assign(self, sess, model):
        sess.run(model.lr.assign(self.lr))

    def setZero(self, sess, model):
        sess.run(model.lr.assign(0.))

class SampleRateUpdater(object):

    def __init__(self, init_sr, num_steps, coef=2.):
        self.sr = init_sr
        self.update_steps = num_steps / 50
        self.coef = coef

    def update(self, global_step):
        if global_step % self.update_steps == 0:
            self.sr /= self.coef
            return True

    def assign(self, sess, model, val=None):
        if val is None: val = self.sr
        sess.run(tf.assign(model.sample_rate, tf.constant(val, dtype=tf.float32)))

class SummaryWriter(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def write_summaries(self, summ, global_step):
        for s in summ:
            self.writer.add_summary(s, global_step)

    def flush(self):
        self.writer.flush()