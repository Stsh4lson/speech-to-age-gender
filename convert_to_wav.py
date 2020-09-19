from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
import os

"""
DONT USE IT WILL WEIGHT 160GB ;(
"""

train = pd.read_csv('data/en/train.tsv', sep='\t')
train = train[['client_id', 'path', 'age', 'gender', 'sentence']].dropna()
train = train[train['gender'] != 'other']

for folder_name in ['data_wav', 'data_wav\\clips']:
    try:
        os.mkdir(folder_name)
        print("Directory", folder_name,  "created ")
    except FileExistsError:
        print("Directory", folder_name,  "already exists")

for case_num in tqdm(range(train.shape[0])):
    path = str('data/en/clips/' + train['path'].iloc[case_num])
    sound = AudioSegment.from_mp3(path)
    sound.export("data_wav/clips/{}.wav".format(train['path'].iloc[case_num])[:-4],
                 format="wav")
