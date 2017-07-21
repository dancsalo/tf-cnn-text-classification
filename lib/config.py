import os.path as osp
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.SAVE_DIRECTORY = 'logs/'
__C.MODEL_DIRECTORY = 'text_conv/'
__C.DISPLAY_STEP = 500
__C.SEED = 123


################################################
# Training Parameters
################################################

__C.TRAIN = edict()

__C.TRAIN.LEARN_RATE = 0.001
__C.TRAIN.WEIGHT_DECAY = 0.0001
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_EPOCHS = 200
__C.TRAIN.KEEP_PROB = 0.5

################################################
# Network Parameters
################################################

__C.NETWORK = edict()

# Convolutional Model Parameters
__C.NETWORK.FILTER_SIZES = [3, 4, 5]
__C.NETWORK.EMBED_SIZE = 128
__C.NETWORK.NUM_FILTERS = 128

# VAE Model Parameters
__C.NETWORK.HIDDEN_SIZE = 128

################################################
# Data Parameters
################################################

__C.DATA = edict()

# Tweet Sentiment Parameters
__C.DATA.POS_FILE = "./data/rt-polaritydata/rt-polarity.pos"
__C.DATA.NEG_FILE = "./data/rt-polaritydata/rt-polarity.neg"
__C.DATA.NUM_CLASSES = 2
