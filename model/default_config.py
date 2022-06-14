from yacs.config import CfgNode as CN

_C = CN()
_C.MODE = 'train'
##########################################################################
## data
## for data config setting
##########################################################################
_C.DATA = CN()
_C.DATA.TRAIN_FILE = './data/train.csv'
_C.DATA.TEST_FILE = './data/test.csv'
_C.DATA.ATTR_FILE = './data/attr.csv'
_C.DATA.RESULT_PATH = './data/result.txt'

_C.DATA.ATTR1_ID2IDX = './data/attr1_id2idx.json'
_C.DATA.IDX2ATTR1_ID = './data/idx2attr1_id.json'
_C.DATA.ATTR2_ID2IDX = './data/attr2_id2idx.json'
_C.DATA.IDX2ATTR2_ID = './data/idx2attr2_id.json'
_C.DATA.USER_ID2IDX = './data/user_id2idx.json'
_C.DATA.IDX2USER_ID = './data/idx2user_id.json'
_C.DATA.ITEM_ID2IDX = './data/item_id2idx.json'
_C.DATA.IDX2ITEM_ID = './data/idx2item_id.json'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

##########################################################################
## model 
## for model config setting
##########################################################################
_C.MODEL = CN()

## RS 
_C.MODEL.DEVICE = 'cuda:0'
_C.MODEL.DROPOUT_RATE1 = 0.15
_C.MODEL.DROPOUT_RATE2 = 0.20
_C.MODEL.USER_NUM = 19835
_C.MODEL.ITEM_NUM = 602305
_C.MODEL.ATTR1_NUM = 52188 + 1  # including the -1 for the padding
_C.MODEL.ATTR2_NUM = 19692 + 1 # including the -1 for the padding
_C.MODEL.EMBEDDING_DIM = 512
_C.MODEL.PARA1 = 1024 
_C.MODEL.PARA2 = 512     
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


##########################################################################
## train
## for train config setting
##########################################################################
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 200
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.VALID_RATE = 0.1
_C.TRAIN.SAVE_EVERY = 1
_C.TRAIN.SAVE_PATH = './checkpoints/'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


##########################################################################
## test
## for test config setting
##########################################################################
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128

#*************************************************************************


# export config
def get_cfg_defaults():
    return _C.clone()

cfg = get_cfg_defaults()