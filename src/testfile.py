import tensorflow as tf

class Saver:
    def __init__(self, test):
        self.checkpoint = tf.train.get_checkpoint_state("checkpoint")