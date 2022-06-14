import torch.nn as nn
import torch 

class RS(nn.Module):
    def __init__(self, cfg, embedding_dim, para1, para2, attr):
        '''
        @param cfg: config options
        @param embedding_dim: the dimension of embedding
        @param para1: the parameter of the first layer
        @param para2: the parameter of the second layer
        @param attr: the attribute of the item
        '''
        super(RS, self).__init__()
        self.cfg = cfg
        self.embedding_dim = embedding_dim
        self.para1 = para1
        self.para2 = para2
        self.attr = attr
        
        self.default1 = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.default2 = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.default3 = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.default4 = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        
        self.embedding_user = nn.Embedding(self.cfg.MODEL.USER_NUM, self.embedding_dim)
        self.embedding_item = nn.Embedding(self.cfg.MODEL.ITEM_NUM, self.embedding_dim)
        self.embedding_attr1 = nn.Embedding(self.cfg.MODEL.ATTR1_NUM, self.embedding_dim)
        self.embedding_attr2 = nn.Embedding(self.cfg.MODEL.ATTR2_NUM, self.embedding_dim)

        
        self.CF_layer1 = nn.Linear(self.embedding_dim*4, self.para1)
        self.BN_layer1 = nn.BatchNorm1d(self.para1)
        self.act_layer1  = nn.ReLU()
        self.Dropout_layer1 = nn.Dropout(self.cfg.MODEL.DROPOUT_RATE1)
        
        self.CF_layer2 = nn.Linear(self.para1, self.para2)
        self.BN_layer2 = nn.BatchNorm1d(self.para2)
        self.act_layer2  = nn.ReLU()
        self.Dropout_layer2 = nn.Dropout(self.cfg.MODEL.DROPOUT_RATE2)
        
        self.CF_layer3 = nn.Linear(self.para2, 1)
    
    def forward(self, device, user_id, item_id, attr1_id, attr2_id):
        # handle the special token -1 in the input
        for i in range(attr1_id.size()[0]):
            # if attr1_id[i] == 0:        # None 
                # attr1_id[i] = self.default1
            if attr1_id[i] == -1:       # Not exist attr1
                attr1_id[i] = self.default3
            # if attr2_id[i] == 0:        # None
            #     attr2_id[i] = self.default2
            if attr2_id[i] == -1:       # Not exist attr2
                attr2_id[i] = self.default4

        attr1 = self.embedding_attr1(attr1_id.long())
        attr2 = self.embedding_attr2(attr2_id.long())
        user = self.embedding_user(user_id)
        item = self.embedding_item(item_id)

        attr1.squeeze_()
        attr2.squeeze_()
        user.squeeze_()
        item.squeeze_()

        x = torch.cat((user, item, attr1, attr2), 1)
        x = x.to(device)

        x = self.CF_layer1(x)
        x = self.BN_layer1(x)
        x = self.act_layer1(x)
        x = self.Dropout_layer1(x)

        x = self.CF_layer2(x)
        x = self.BN_layer2(x)
        x = self.act_layer2(x)
        x = self.Dropout_layer2(x)

        x = self.CF_layer3(x)

        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))