import pandas as pd

def read_data(file_name, data_type = 'train'):
    assert data_type in ['train', 'test', 'item'], 'data_type must be train, test or item'
    data = open(file_name, 'r')
    
    if data_type == 'train':
        users = []
        items = []
        ratings = []
        for each_line in data:
            if not each_line.find('|') == -1:
                (user_id, item_num) = each_line.strip('\n').split('|')
                for i in range(int(item_num)):
                    users.append(int(user_id))
            else:
                (item_id, rating) = each_line.strip('\n').split('  ')
                items.append(int(item_id))
                ratings.append(int(rating))
        return users, items, ratings
    elif data_type == 'test':
        users = []
        items = []
        for each_line in data:
            if not each_line.find('|') == -1:
                (user_id, item_num) = each_line.strip('\n').split('|')
                for i in range(int(item_num)):
                    users.append(int(user_id))
            else:
                (item_id) = each_line
                items.append(int(item_id))
        return users, items
    elif data_type == 'item':
        items = []
        attrs = []
        for each_line in data:
            (item_id, attr1, attr2) = each_line.strip('\n').split('|')
            items.append(int(item_id))
            attrs.append([int(attr1 if attr1 != 'None' else 0), int(attr2 if attr2 != 'None' else 0)])
        return items, attrs

    return data


def get_attr_dict(file_name):
    attr_dict = {}
    attrs = pd.read_csv(file_name)
    for index, row in attrs.iterrows():
        attr_dict[row['item_id']] = [row['attr1'], row['attr2']]
    return attr_dict

if __name__ == "__main__":
    pass 