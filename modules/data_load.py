from sklearn.model_selection import train_test_split
import pandas as pd


def data_load(clip=5000):
    """
    -Load data from train and test.tsv
    -clipping to set amount of samples
    -spliting to train, val, test
    -sampling unique speakers for each dataset
    -saving to data_info.csv
    """
    train = pd.read_csv('data/en/train.tsv', sep='\t')
    train = train[['client_id', 'path', 'age', 'gender', 'sentence']].dropna(
        subset=['age', 'gender'], how='any')
    train = train[train.gender != 'other']
    train = train[train.age != 'nineties']

    # sampling classes up to 'clip' if there is more
    def clip_classes(train, clip):
        train_temp = pd.DataFrame()
        for gender_class in train.gender.unique():
            for age_class in train.age.unique():
                subset = train[(train.age == age_class) &
                            (train.gender == gender_class)]
                if subset.shape[0] >= clip:
                    subset = subset.sample(clip, random_state=795797950)
                train_temp = pd.concat([
                    train_temp,
                    subset
                ])
        return train_temp.reset_index(drop=True)

    train = clip_classes(train, clip)

    train_unique = train.drop_duplicates(subset=['client_id'])
    train_clients, val_clients = train_test_split(
        train_unique, test_size=0.05, random_state=795797950)
    train_clients = train[train.client_id.isin(train_clients.client_id)].copy()
    val_clients = train[train.client_id.isin(val_clients.client_id)].copy()
    train_clients['set_type'] = 'train'
    val_clients['set_type'] = 'val'
    train = pd.concat([train_clients, val_clients])

    test = pd.read_csv('data/en/test.tsv', sep='\t')
    test = test[['client_id', 'path', 'age', 'gender', 'sentence']].dropna(
        subset=['age', 'gender'], how='any')
    test = test[test.gender != 'other']
    test = test[test.age != 'nineties']
    test['set_type'] = 'test'
    data_df = pd.concat([train, test])

    print('Train set distribution:')
    print(train_clients.groupby(['age', 'gender']).size())
    print('Val set distribution:')
    print(val_clients.groupby(['age', 'gender']).size())
    print('Test set distribution:')
    print(test.groupby(['age', 'gender']).size())

    overlap = (train_clients[train_clients['client_id'].isin(
        test['client_id'])].shape[0] +
        train_clients[train_clients['client_id'].isin(
            val_clients['client_id'])].shape[0])
    if overlap == 0:
        print("Unique speakers in each set")

    data_df.to_csv('data_info.csv', index=False)
