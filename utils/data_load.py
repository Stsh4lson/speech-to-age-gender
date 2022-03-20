from sklearn.model_selection import train_test_split
import pandas as pd


def data_load():
    test = pd.read_csv('data/en/test.tsv', sep='\t')
    test['set_type'] = 'test'
    train = pd.read_csv('data/en/train.tsv', sep='\t')
    train['set_type'] = 'train'
    data = pd.concat((train, test))
    data = data[['client_id', 'path', 'age', 'gender', 'sentence', 'set_type']]
    data = data.dropna(subset=['age', 'gender'], how='any')
    data = data[data.gender != 'other']
    data = data[data.age != 'nineties']
    data = data[data.age != 'eighties']
    data.to_csv('data_info.csv', index=False)
