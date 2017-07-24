# --------------------------------------------------------
# CNN Text Classification
# Copyright (c) 2017 Automated Insights
# Written by Dan Salo
# --------------------------------------------------------

from tensorflow.contrib import learn

import numpy as np
import re


class RTPolarity:
    def __init__(self, flags):
        self.name = 'RTPolarity'
        self.flags = flags
        self.train_data, self.test_data, self.seq_length, self.vocab_size, self.vocab_dict = self._load_data()
        self.num_train_examples = len(self.train_data)
        self.num_test_examples = len(self.test_data)
        self.num_batches_per_epoch = int((self.num_train_examples - 1) / self.flags['TRAIN']['BATCH_SIZE']) + 1

    def _load_data(self):
        """
        Load the data, preprocess, and return to initializing function
        """
        x_text, y = self._load_data_and_labels(self.flags['DATA'][self.name]['POS_FILE'], self.flags['DATA'][self.name][
            'NEG_FILE'])

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)
        vocab_dict = {v: k for k, v in vocab_processor.vocabulary_._mapping.items()}

        # Randomly shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        dev_percent = 0.15
        dev_sample_index = -1 * int(dev_percent * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        seq_length = x_train.shape[1]

        train_data = list(zip(x_train, y_train))
        test_data = list(zip(x_dev, y_dev))
        return train_data, test_data, seq_length, vocab_size, vocab_dict

    def batch_iter(self, batch_size, num_epochs, mode, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        if mode == 'train':
            data = np.array(self.train_data)
        else:  # mode == 'test
            data = np.array(self.test_data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def _load_data_and_labels(self, positive_data_file, negative_data_file):
        """
        Loads Rotten Tomates polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self._clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def transform_x(self, x):
        """
        Transforms an array of numbers that correspond to words in the vocab into those words
        """
        x_text = [self.vocab_dict[number] for number in x if number != 0]
        return ' '.join(x_text)

    @staticmethod
    def transform_y(y):
        """
        Transforms an array of labels the correspond to the data categories and returns that category
        """
        y_label = np.argmax(y)
        if y_label == 0:
            return 'negative'
        else:  # y_label == 1
            return 'positive'

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


class RTImdbSentiment:
    def __init__(self, flags):
        self.name = 'RTImdbSentiment'
        self.flags = flags
        self.train_data, self.test_data, self.seq_length, self.vocab_size, self.vocab_dict = self._load_data()
        self.num_train_examples = len(self.train_data)
        self.num_test_examples = len(self.test_data)
        self.num_batches_per_epoch = int((self.num_train_examples - 1) / self.flags['TRAIN']['BATCH_SIZE']) + 1

    def _load_data(self):
        """
        Load the data, preprocess, and return to initializing function
        """
        x_text, y = self._load_data_and_labels(self.flags['DATA'][self.name]['OBJ_FILE'], self.flags['DATA'][self.name][
            'SUBJ_FILE'])

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)
        vocab_dict = {v: k for k, v in vocab_processor.vocabulary_._mapping.items()}

        # Randomly shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        dev_percent = 0.15
        dev_sample_index = -1 * int(dev_percent * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        seq_length = x_train.shape[1]

        train_data = list(zip(x_train, y_train))
        test_data = list(zip(x_dev, y_dev))
        return train_data, test_data, seq_length, vocab_size, vocab_dict

    def batch_iter(self, batch_size, num_epochs, mode, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        if mode == 'train':
            data = np.array(self.train_data)
        else:  # mode == 'test
            data = np.array(self.test_data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def _load_data_and_labels(self, objective_data_file, subjective_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        objective_examples = list(open(objective_data_file, "r", encoding='latin-1').readlines())
        objective_examples = [s.strip() for s in objective_examples]
        subjective_examples = list(open(subjective_data_file, "r", encoding='latin-1').readlines())
        subjective_examples = [s.strip() for s in subjective_examples]
        # Split by words
        x_text = objective_examples + subjective_examples
        x_text = [self._clean_str(sent) for sent in x_text]
        # Generate labels
        objective_labels = [[0, 1] for _ in objective_examples]
        subjective_labels = [[1, 0] for _ in subjective_examples]
        y = np.concatenate([objective_labels, subjective_labels], 0)
        return [x_text, y]

    def transform_x(self, x):
        """
        Transforms an array of numbers that correspond to words in the vocab into those words
        """
        x_text = [self.vocab_dict[number] for number in x if number != 0]
        return ' '.join(x_text)

    @staticmethod
    def transform_y(y):
        """
        Transforms an array of labels the correspond to the data categories and returns that category
        """
        y_label = np.argmax(y)
        if y_label == 0:
            return 'subjective'
        else:  # y_label == 1
            return 'objective'

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()