# --------------------------------------------------------
# CNN Text Classification
# Copyright (c) 2017 Automated Insights
# Written by Dan Salo
# --------------------------------------------------------

import tensorflow as tf
import numpy as np
import logging
import os

from tensorflow.python import pywrap_tensorflow
import sys


class Model:
    """
    A Class for easy Model Training. Taken from TensorBase modele. dancsalo@github.
    Methods:
        See list in __init__() function
    """

    def __init__(self, flags, config_dict=None):
        config_yaml_flags_dict = self.load_config_yaml(flags, config_dict)
        config_yaml_flags_dict_none = self.check_dict_keys(config_yaml_flags_dict)

        print(config_yaml_flags_dict_none)
        # Define constants
        self.step = 1
        self.flags = config_yaml_flags_dict_none

        # Run initialization functions
        self._check_file_io()
        self._data()
        self._set_seed()
        self._network()
        self._optimizer()
        self._summaries()
        self.merged, self.saver, self.sess, self.writer = self._set_tf_functions()
        self._initialize_model()

    def load_config_yaml(self, flags, config_dict):
        """ Load config dict and yaml dict and then override both with flags dict. """
        if config_dict is None:
            print('Config File not specified. Using only input flags.')
            return flags
        try:
            config_yaml_dict = self.cfg_from_file(flags['YAML_FILE'], config_dict)
        except KeyError:
            print('Yaml File not specified. Using only input flags and config file.')
            return config_dict
        print('Using input flags, config file, and yaml file.')
        config_yaml_flags_dict = self._merge_a_into_b_simple(flags, config_yaml_dict)
        return config_yaml_flags_dict

    def check_dict_keys(self, config_yaml_flags_dict):
        """ Fill in all optional keys with None. Exit in a crucial key is not defined. """
        crucial_keys = ['MODEL_DIRECTORY', 'SAVE_DIRECTORY']
        for key in crucial_keys:
            if key not in config_yaml_flags_dict:
                print('You must define %s. Now exiting...' % key)
                exit()
        optional_keys = ['RESTORE_SLIM_FILE', 'RESTORE_META', 'RESTORE_SLIM', 'SEED', 'GPU']
        for key in optional_keys:
            if key not in config_yaml_flags_dict:
                config_yaml_flags_dict[key] = None
                print('%s in flags, yaml or config dictionary was not found.' % key)
        if 'RUN_NUM' not in config_yaml_flags_dict:
            config_yaml_flags_dict['RUN_NUM'] = 0
        return config_yaml_flags_dict

    def _check_file_io(self):
        """ Create and define logging directory """
        folder = 'Model' + str(self.flags['RUN_NUM']) + '/'
        folder_restore = 'Model' + str(self.flags['MODEL_RESTORE']) + '/'
        print(self.flags['DATASET'])
        self.flags['RESTORE_DIRECTORY'] = self.flags['SAVE_DIRECTORY'] + self.flags[
            'MODEL_DIRECTORY'] + self.flags['DATASET_NAME'] + '/' + folder_restore
        self.flags['LOGGING_DIRECTORY'] = self.flags['SAVE_DIRECTORY'] + self.flags[
            'MODEL_DIRECTORY'] + self.flags['DATASET_NAME'] + '/' + folder
        self.make_directory(self.flags['LOGGING_DIRECTORY'])
        sys.stdout = Logger(self.flags['LOGGING_DIRECTORY'] + 'ModelInformation.log')
        print(self.flags)

    def _set_tf_functions(self):
        """ Sets up summary writer, saver, and session, with configurable gpu visibility """
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        if type(self.flags['GPU']) is int:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.flags['GPU'])
            print('Using GPU %d' % self.flags['GPU'])
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        sess = tf.Session(config=config)
        writer = tf.summary.FileWriter(self.flags['LOGGING_DIRECTORY'], sess.graph)
        return merged, saver, sess, writer

    def _get_restore_meta_file(self):
        return 'part_' + str(self.flags['FILE_EPOCH']) + '.ckpt.meta'

    def _restore_meta(self):
        """ Restore from meta file. 'RESTORE_META_FILE' is expected to have .meta at the end. """
        restore_meta_file = self._get_restore_meta_file()
        filename = self.flags['RESTORE_DIRECTORY'] + self._get_restore_meta_file()
        new_saver = tf.train.import_meta_graph(filename)
        new_saver.restore(self.sess, filename[:-5])
        print("Model restored from %s" % restore_meta_file)

    def _restore_slim(self, variables):
        """ Restore from tf-slim file (usually a ImageNet pre-trained model). """
        variables_to_restore = self.get_variables_in_checkpoint_file(self.flags['RESTORE_SLIM_FILE'])
        variables_to_restore = {self.name_in_checkpoint(v): v for v in variables if (self.name_in_checkpoint(v) in variables_to_restore)}
        if variables_to_restore is []:
            print('Check the SLIM checkpoint filename. No model variables matched the checkpoint variables.')
            exit()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, self.flags['RESTORE_SLIM_FILE'])
        print("Model restored from %s" % self.flags['RESTORE_SLIM_FILE'])

    def _initialize_model(self):
        """ Initialize the defined network and restore from files is so specified. """
        # Initialize all variables first
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        if self.flags['RESTORE_META'] == 1:
            print('Restoring from .meta file')
            self._restore_meta()
        elif self.flags['RESTORE_SLIM'] == 1:
            print('Restoring TF-Slim Model.')
            all_model_variables = tf.global_variables()
            self._restore_slim(all_model_variables)
        else:
            print("Model training from scratch.")

    def _init_uninit_vars(self):
        """ Initialize all other trainable variables, i.e. those which are uninitialized """
        uninit_vars = self.sess.run(tf.report_uninitialized_variables())
        vars_list = list()
        for v in uninit_vars:
            var = v.decode("utf-8")
            vars_list.append(var)
        uninit_vars_tf = [v for v in tf.global_variables() if v.name.split(':')[0] in vars_list]
        self.sess.run(tf.variables_initializer(var_list=uninit_vars_tf))

    def _save_model(self, section):
        """ Save model in the logging directory """
        checkpoint_name = self.flags['LOGGING_DIRECTORY'] + 'part_%d' % section + '.ckpt'
        save_path = self.saver.save(self.sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)

    def _record_training_step(self, summary):
        """ Adds summary to writer and increments the step. """
        self.writer.add_summary(summary=summary, global_step=self.step)
        self.step += 1

    def _set_seed(self):
        """ Set random seed for numpy and tensorflow packages """
        if self.flags['SEED'] is not None:
            tf.set_random_seed(self.flags['SEED'])
            np.random.seed(self.flags['SEED'])

    def _summaries(self):
        """ Print out summaries for every variable. Can be overriden in main function. """
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            print(var.name)

    def _data(self):
        """Define data"""
        raise NotImplementedError

    def _network(self):
        """Define network"""
        raise NotImplementedError

    def _optimizer(self):
        """Define optimizer"""
        raise NotImplementedError

    def get_flags(self):
        return self.flags

    @staticmethod
    def make_directory(folder_path):
        """ Make directory at folder_path if it does not exist """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def print_log(message):
        """ Print message to terminal and to logging document if applicable """
        print(message)
        logging.info(message)

    @staticmethod
    def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)

    @staticmethod
    def name_in_checkpoint(var):
        """ Removes 'model' scoping if it is present in order to properly restore weights. """
        if var.op.name.startswith('model/'):
            return var.op.name[len('model/'):]

    @staticmethod
    def get_variables_in_checkpoint_file(filename):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(filename)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def _merge_a_into_b(self, a, b):
        """Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a.
        """
        from easydict import EasyDict as edict
        if type(a) is not edict:
            return

        for k, v in a.items():
            # a must specify keys that are in b
            if k not in b:
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(type(b[k]),
                                                                   type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    self._merge_a_into_b(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                b[k] = v
        return b

    def _merge_a_into_b_simple(self, a, b):
        """Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a. Do not do any checking.
        """
        for k, v in a.items():
            b[k] = v
        return b

    def cfg_from_file(self, yaml_filename, config_dict):
        """Load a config file and merge it into the default options."""
        import yaml
        from easydict import EasyDict as edict
        with open(yaml_filename, 'r') as f:
            yaml_cfg = edict(yaml.load(f))

        return self._merge_a_into_b(yaml_cfg, config_dict)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
