import torch.utils.data as data
import pandas as pd
import torch
import json

class Dataset(data.Dataset):
    def __init__(self, cfg, filename, attr, type,validate_rate = None):
        '''
        :param cfg: config options
        :param filename: the path of the dataset
        :param attr: the path of the attribute
        :param type: train, test or valid
        :param validate_rate: the rate of validation
        '''
        super(Dataset, self).__init__()
        self.frame = pd.read_csv(filename)
        self.attr = attr
        self.type = type
        
        self.attr1_id2idx = json.load(open(cfg.DATA.ATTR1_ID2IDX, 'r'))
        self.idx2attr1_id = json.load(open(cfg.DATA.IDX2ATTR1_ID, 'r'))
        self.attr2_id2idx = json.load(open(cfg.DATA.ATTR2_ID2IDX, 'r'))
        self.idx2attr2_id = json.load(open(cfg.DATA.IDX2ATTR2_ID, 'r'))
        self.user_id2idx = json.load(open(cfg.DATA.USER_ID2IDX, 'r'))
        self.idx2user_id = json.load(open(cfg.DATA.IDX2USER_ID, 'r'))
        self.item_id2idx = json.load(open(cfg.DATA.ITEM_ID2IDX, 'r'))
        self.idx2item_id = json.load(open(cfg.DATA.IDX2ITEM_ID, 'r'))
        
        if validate_rate is not None:
            if type == 'train':
                self.frame = self.frame.iloc[:int(len(self.frame) * (1 - validate_rate))]
                print('train set size:', len(self.frame))
            elif type == 'valid':
                self.frame = self.frame.iloc[int(len(self.frame) * (1 - validate_rate)):]
                print('valid set size:', len(self.frame))
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, index):
        if self.type == 'train' or self.type == 'valid':
            user, item, target = self.frame.iloc[index, 0], self.frame.iloc[index, 1], self.frame.iloc[index, 2]
        elif self.type == 'test':
            user, item  = self.frame.iloc[index, 0], self.frame.iloc[index, 1]
        user_idx = int(self.user_id2idx[str(user)])
        item_idx = int(self.item_id2idx[str(item)])
        try:
            attr1, attr2 = self.attr[item][0], self.attr[item][1]
            attr1_idx =  int(self.attr1_id2idx[str(attr1)])
            attr2_idx =  int(self.attr2_id2idx[str(attr2)])
        except KeyError:
            attr1_idx, attr2_idx = -1, -1  # -1 means no attribute
        attr1_idx, attr2_idx = torch.FloatTensor([attr1_idx]), torch.FloatTensor([attr2_idx])
        if self.type == 'train' or self.type == 'valid':
            return user_idx, item_idx, target, attr1_idx, attr2_idx
        elif self.type == 'test':
            return user_idx, item_idx, attr1_idx, attr2_idx

