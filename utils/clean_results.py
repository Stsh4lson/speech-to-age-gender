import os
import shutil
from tqdm import tqdm

'''
Removes results from experiments that have less than 2 epochs

'''

empty_results = []
for folder in os.listdir('results'):
    directory = "/".join(('results', folder))
    if not os.path.exists(directory + '/model/checkpoint/checkpoint.h5'):
        empty_results.append(directory)
    # print(f'removed {directory}')
    # shutil.rmtree(directory)

print(f'{len(empty_results)} non complete folders')
if len(empty_results) > 0:
    for f in tqdm(empty_results):
        shutil.rmtree(f)