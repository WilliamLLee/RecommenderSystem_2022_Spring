from model.utils import *
import json
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np

train_data = read_data('./data/train.txt', 'train')
print("训练数据(user, item, rating)数量：", len(train_data[0]))
# print(train_data[0][0:10], train_data[1][0:10], train_data[2][0:10])
train_record = np.column_stack((train_data[0], train_data[1], train_data[2]))
print("训练数据规模: ", train_record.shape)
df_train = pd.DataFrame(train_record, columns=['user_id', 'item_id', 'rating'], dtype=int)
print("训练集 user_id, item_id 的数量：", df_train['user_id'].nunique(), df_train['item_id'].nunique())
df_train = df_train.drop_duplicates(subset=['user_id', 'item_id'])
print("去除重复的user, item 对后数据集规模：", df_train.shape)
print("训练数据集评分情况统计1: \n", df_train['rating'].describe())
print("训练数据集评分情况统计2: \n", df_train['rating'].value_counts(sort=True, ascending=False, bins = 10))
# print("训练数据集评分情况统计3: \n", df_train['rating'].value_counts(sort=True).plot(kind='bar'))
# plt.show()

# # show the data score distribution
# sns.distplot(df_train['rating'])
# # add title and save the figure
# plt.title('Rating Distribution')
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.savefig('./files/rating_distribution.png')
# plt.show()

print('*'*50)
test_data = read_data('./data/test.txt', 'test')
print('测试数据(user, item)数量：',len(test_data[0]))
# print(test_data[0][0:10], test_data[1][0:10])
test_record = np.column_stack((test_data[0], test_data[1]))
print("测试数据集规模：", test_record.shape)
df_test = pd.DataFrame(test_record, columns=['user_id', 'item_id'])
print("测试集 user_id, item_id 的数量：", df_test['user_id'].nunique(), df_test['item_id'].nunique())
df_test = df_test.drop_duplicates(subset=['user_id', 'item_id'])
print("去重后测试数据集规模：", df_test.shape)

print('*'*50)
data = read_data('./data/itemAttribute.txt', 'item')
print('属性数据(item, attrs)数量：', len(data[0]))
# print(data[0][0:10], data[1][0:10])
attr_record = np.column_stack((data[0], data[1]))
print("属性数据集规模：", attr_record.shape)
df_attr = pd.DataFrame(attr_record, columns=['item_id', 'attr1', 'attr2'])
print("属性1的数量：", df_attr['attr1'].nunique())
# 构建属性1的映射并保存到文件
attr1_id2idx = {str(v): str(k) for k, v in enumerate(df_attr['attr1'].unique())}
idx2attr2_id = {str(k): str(v) for k, v in enumerate(df_attr['attr1'].unique())}
with open('./data/attr1_id2idx.json', 'w') as f:
    json.dump(attr1_id2idx, f)
f.close()
with open('./data/idx2attr1_id.json', 'w') as f:
    json.dump(idx2attr2_id, f)
f.close()
# 构建属性2的映射并保存到文件
attr2_id2idx = {str(v): str(k) for k, v in enumerate(df_attr['attr2'].unique())}
idx2attr2_id = {str(k): str(v) for k, v in enumerate(df_attr['attr2'].unique())}
with open('./data/attr2_id2idx.json', 'w') as f:
    json.dump(attr2_id2idx, f)
f.close()
with open('./data/idx2attr2_id.json', 'w') as f:
    json.dump(idx2attr2_id, f)
f.close()

print("属性2的数量：", df_attr['attr2'].nunique())
df_attr = df_attr.drop_duplicates(subset=['item_id'])
print("去重后属性数据集规模：", df_attr.shape)



print('*'*50)
# 训练集与测试集user_id的并集和交集数量
print("训练集与测试集user_id的并集数量：", len(set(df_train['user_id']) | set(df_test['user_id'])))
# 基于训练集和测试集user_id的并集来构建user_id的映射
user_id2idx = {str(id): str(i) for i, id in enumerate(set(df_train['user_id']) | set(df_test['user_id']))}
idx2user_id = {str(i): str(id) for i, id in enumerate(set(df_train['user_id']) | set(df_test['user_id']))}

# 保存映射到文件
with open('./data/user_id2idx.json', 'w') as f:
    json.dump(user_id2idx, f)
f.close()
with open('./data/idx2user_id.json', 'w') as f:
    json.dump(idx2user_id, f)
f.close()

print("训练集与测试集user_id的交集数量：", len(set(df_train['user_id']) & set(df_test['user_id'])))
# 训练集测试集的item_id的并集和交集数量
print("训练集与测试集item_id的并集数量：", len(set(df_train['item_id']) | set(df_test['item_id'])))
print("训练集与测试集item_id的交集数量：", len(set(df_train['item_id']) & set(df_test['item_id'])))

# 训练集、测试集、属性集中item_id的并集和交集数量
print("训练集、测试集、属性集中item_id的并集数量：", len(set(df_train['item_id']) | set(df_test['item_id']) | set(df_attr['item_id'])))
print("训练集、测试集、属性集中item_id的交集数量：", len(set(df_train['item_id']) & set(df_test['item_id']) & set(df_attr['item_id'])))

# 基于训练集、测试集、属性集中item_id的并集构建item_id的映射
item_id2idx = {str(id): str(i) for i, id in enumerate(set(df_train['item_id']) | set(df_test['item_id']) | set(df_attr['item_id']))}
idx2item_id = {str(i): str(id) for i, id in enumerate(set(df_train['item_id']) | set(df_test['item_id']) | set(df_attr['item_id']))}
# 保存映射到文件
with open('./data/item_id2idx.json', 'w') as f:
    json.dump(item_id2idx, f)
f.close()
with open('./data/idx2item_id.json', 'w') as f:
    json.dump(idx2item_id, f)
f.close()


print('*'*50)
print("保存训练数据至{} ...".format('./files/train.csv'))
# shuffle the train data for better training and validation
df_train = df_train.sample(frac=1)
df_train.to_csv('./data/train.csv', index=False)
print("保存完毕！")
print("保存测试数据至{} ...".format('./files/test.csv'))
df_test.to_csv('./data/test.csv', index=False)
print("保存完毕！")
print("保存属性数据至{} ...".format('./files/attr.csv'))
df_attr.to_csv('./data/attr.csv', index=False)
print("保存完毕！")