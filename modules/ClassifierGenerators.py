import numpy as np
import os
import pandas as pd


def _generator(SET_TYPE):
    metadata = pd.read_csv(os.path.join('data', 'images',
                            'image_metadata.csv'))
    metadata = metadata[metadata.set_type == 'train']
    metadata.age = metadata.age.map({'teens': 0, 'twenties': 1,
                                     'seventies': 2, 'fifties': 3,
                                     'fourties': 4, 'thirties': 5,
                                     'sixties': 6, 'eighties': 7}
                                     )
    metadata.gender = metadata.gender.map({'male': 0, 'female': 1})
    metadata = metadata.sample(frac=-1)
    L = metadata.shape[0]
    for sample in range(L):
        X = np.expand_dims(np.load(metadata.iloc[sample, 0]), axis=-1)
        y_gender = metadata.iloc[sample, 2]
        y_age = metadata.iloc[sample, 3]
        yield X, y_gender, y_age