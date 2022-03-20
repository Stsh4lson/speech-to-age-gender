import pygame
import pandas as pd
import os


class PlaySample():
    def __init__(self):
        cwd = os.getcwd()
        if os.path.split(cwd)[1] == 'utils':
            os.chdir(os.path.dirname(cwd))
        self.metadata = pd.read_csv('data_info.csv')
        self.metadata = self.metadata[self.metadata['set_type'] == 'train']
        self.og_metadata_train = pd.read_csv(os.path.join('data',
                                                          'en',
                                                          'train.tsv'), sep='\t')

    def play_random(self):
        random_sample = self.metadata.sample(1)
        sample_path = random_sample['path'].values[0]
        sample_path_full = os.path.join(os.getcwd(), 'data', 'en', 'clips', sample_path)
        print(f'File path: {sample_path_full}')
        sample_labels = random_sample[['age', 'gender']]

        print(f'Age: {sample_labels["age"].values[0]}')
        print(f'Gender: {sample_labels["gender"].values[0]}')
        pygame.mixer.init()
        pygame.mixer.music.load(sample_path_full)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


if __name__ == '__main__':
    ps = PlaySample()
    ps.play_random()
    
