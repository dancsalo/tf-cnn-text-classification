# --------------------------------------------------------
# CNN Text Classification
# Copyright (c) 2017 Automated Insights
# Written by Dan Salo
# --------------------------------------------------------

import argparse
import tensorflow as tf

from importlib import import_module

from lib.model import Model
from lib.config import cfg


class TextClassification(Model):
    def __init__(self, flags_input, cfg):
        flags_input['DATASET_NAME'] = flags_input['DATASET']
        flags_input['DATASET'] = getattr(import_module('lib.datasets'), flags_input['DATASET'])
        flags_input['ARCHITECTURE'] = getattr(import_module('lib.networks'), flags_input['ARCHITECTURE'])
        super().__init__(flags_input, cfg)

    def _summaries(self):
        """ Define summaries for Tensorboard """
        self.Network.summaries()

    def _data(self):
        """ Define data I/O """
        self.Data = self.flags['DATASET'](self.flags)

    def _network(self):
        """ Define network """
        self.Network = self.flags['ARCHITECTURE'](self.flags, self.Data)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        self.optimizer = tf.train.AdamOptimizer(self.flags['TRAIN']['LEARN_RATE']).minimize(self.Network.cost)

    def _run_train_metrics_iter(self, batch):
        """ Run training iteration and also calculate metrics """
        feed_dict = self.Network.feed_dict(batch, 'train')
        outputs = self.Network.outputs([self.merged, self.optimizer], 'train')
        return self.sess.run(outputs, feed_dict=feed_dict)

    def _run_eval_iter(self, batch):
        feed_dict = self.Network.feed_dict(batch, 'test')
        outputs = self.Network.outputs([], 'test')
        return self.sess.run(outputs, feed_dict=feed_dict)

    def train(self):
        """ Train the model """
        epochs = 0
        batches = self.Data.batch_iter(self.flags['TRAIN']['BATCH_SIZE'], self.flags['TRAIN']['NUM_EPOCHS'], 'train')
        for batch in batches:
            outputs = self._run_train_metrics_iter(batch)
            if self.step % self.flags['DISPLAY_STEP'] != 0:
                output_string = self.Network.record_metrics(outputs, 'train')
                self.print_log(output_string)
            if self.step % self.Data.num_batches_per_epoch == 0:
                epochs += 1
                self._save_model(section=epochs)
            self._record_training_step(summary=outputs[0])

    def test(self):
        """ Evaluate the accuracy of the model on the test set """
        examples = self.Data.batch_iter(batch_size=1, num_epochs=1, mode='test')
        accuracy = list()
        for e in examples:
            if self._run_eval_iter(e)[0] == 0:
                x, y = zip(*e)
                x_text = self.Data.transform_x(x[0])
                y_label = self.Data.transform_y(y[0])
                print('INCORRECT: {}'.format(x_text))
                print('CORRECT LABEL: {}'.format(y_label))
                print('\n')
            accuracy.append(self._run_eval_iter(e)[0])
        print('Total Number of Test Examples: {}'.format(len(accuracy)))
        print('Accuracy Averaged over Test Set: {}'.format(sum(accuracy)/len(accuracy)))


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN Networks Arguments')
    parser.add_argument('-n', '--RUN_NUM', default=0, type=int)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-r', '--RESTORE_META', default=0, type=int)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--MODEL_RESTORE', default=1, type=int)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--FILE_EPOCH', default=1, type=int)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--TRAINING', default=1, type=int)  # Binary to train model. 0 = No train.
    parser.add_argument('-e', '--EVAL', default=1, type=int)  # Binary to evalulate model. 0 = No eval.
    parser.add_argument('-g', '--GPU', default=0, type=int)  # specifiy which GPU to use. Defaults to only one GPU.
    parser.add_argument('-y', '--YAML_FILE', default='cfgs/OneLayerConv.yml', type=str)
    parser.add_argument('-d', '--DATASET', default='RTPolarity', type=str)  # other option, RTImdbSentiment
    parser.add_argument('-a', '--ARCHITECTURE', default='OneLayerConv')
    flags = vars(parser.parse_args())

    model = TextClassification(flags, cfg)
    if flags['TRAINING'] == 1:
        model.train()
    if flags['EVAL'] == 1:
        model.test()


if __name__ == "__main__":
    main()
