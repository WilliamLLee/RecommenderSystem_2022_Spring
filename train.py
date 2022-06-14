import torch
import torch.nn as nn
import os
from tqdm import tqdm 

from model.RS import RS
from model.default_config import cfg
from model.data import Dataset
from model.utils import get_attr_dict


def train(cfg, model, device, train_loader, optimizer):
    model.train()
    loss_fn = nn.MSELoss(reduction='mean')
    total_loss = 0
    with tqdm(total=len(train_loader), desc='Train') as pbar:
        for batch_idx, (user, item, target, attr1, attr2) in enumerate(train_loader):
            user, item, target= user.to(device), item.to(device), target.to(device).float()
            attr1, attr2 = attr1.to(device).int(), attr2.to(device).int()
            optimizer.zero_grad()
            target = target / 10  # normalize the target
            output = model(device, user, item, attr1, attr2)
            loss = loss_fn(output.squeeze(), target)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    return total_loss / len(train_loader)

def test(cfg, model, device, test_loader):
    model.eval()
    result = torch.LongTensor(0, 3)
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Test') as pbar:
            for batch_idx, (user, item, attr1, attr2) in enumerate(test_loader):
                user, item, attr1, attr2 = user.to(device), item.to(device), attr1.to(device).int(), attr2.to(device).int()
                output = model(device, user, item, attr1, attr2)
                result = torch.cat((result, torch.cat((user.unsqueeze(1), item.unsqueeze(1), output.long().unsqueeze(1)), dim=1)), dim=0)
                pbar.update(1)
    # organize the result with user_id categories
    result = result.numpy()
    with open(cfg.DATA.RESULT_PATH, 'w') as f:
        result_dict = {}
        for i in range(len(result)):
            user_id = result[i][0]
            item_id = result[i][1]
            rating = result[i][2]
            if user_id not in result_dict:
                result_dict[user_id] = []
            result_dict[user_id].append((item_id, rating))
        for user_id in result_dict:
            f.write(str(user_id)+'|' + str(len(result_dict[user_id])) + '\n')
            for item_id, rating in result_dict[user_id]:
                f.write(str(item_id) + ' ' + str(rating) + '\n')
    f.close
    return result_dict

def validate(cfg, model, device, validate_loader):
    model.eval()
    total_loss = 0
    errors = []
    with torch.no_grad():
        with tqdm(total=len(validate_loader), desc='Validate') as pbar:
            for batch_idx, (user, item, target, attr1, attr2) in enumerate(validate_loader):
                user, item, attr1, attr2 = user.to(device), item.to(device), attr1.to(device).int(), attr2.to(device).int()
                output = model(device, user, item, attr1, attr2)
                target = target.to(device).float()
                target = target / 10  # normalize the target
                # validate the output with the target
                loss = nn.MSELoss(reduction='mean')(output.squeeze(), target)
                # calculate error and append to errors list
                errors.append(torch.abs(output.squeeze() - target).item())
                total_loss = total_loss + loss.item()
                pbar.update(1)
    # calculate the RMSE and MAE
    rmse = torch.sqrt(torch.mean(torch.tensor(errors)))
    mae = torch.mean(torch.tensor(errors))
    return total_loss / len(validate_loader), rmse, mae


if __name__ == '__main__':
    # set device 
    if cfg.MODEL.DEVICE.startswith('cuda'):
        device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.MODEL.DEVICE)

    attrs = get_attr_dict(cfg.DATA.ATTR_FILE)

    train_dataset = Dataset(cfg, cfg.DATA.TRAIN_FILE, attrs, cfg.MODE, validate_rate = cfg.VALID_RATE)  # cfg.MODE is set to 'train' 
    valid_dataset = Dataset(cfg, cfg.DATA.TRAIN_FILE, attrs, 'valid', validate_rate = cfg.VALID_RATE)
    test_dataset = Dataset(cfg, cfg.DATA.TEST_FILE, attrs, 'test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

    model = RS( cfg, 
                embedding_dim = cfg.MODEL.EMBEDDING_DIM, 
                para1 = cfg.MODEL.PARA1, 
                para2 = cfg.MODEL.PARA2, 
                attr=attrs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        print('Epoch: {} training ...'.format(epoch))
        total_loss = train(cfg, model, device, train_loader, optimizer)
        valida_loss, rmse, mae = validate(cfg, model, device, valid_loader)

        print('Epoch: {}, Train Loss: {}, Validate Loss: {}, RMSE: {}, MAE: {}'.format(epoch + 1, total_loss, valida_loss, rmse, mae))
        if (epoch + 1) % cfg.TRAIN.SAVE_EVERY == 0:
            print('Saving model, epoch: {}'.format(epoch))
            model.save_model(os.path.join(cfg.TRAIN.SAVE_PATH, 'model_epoch_{}.pth'.format(epoch + 1)))

    # test the model
    test(cfg, model, device, test_loader)