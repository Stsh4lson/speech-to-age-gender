import numpy as np
import pandas as pd


def init():
    global AE_TIMESTEPS
    global AE_WINDOWS_STEP
    global AE_N_FEATURES
    global AE_BATCH_SIZE
    global AE_EPOCHS
    global AE_TRAIN_IDX
    global AE_VALIDATION_IDX
    global AE_LATENT_DIM
    global SMALL_DATA_INDICES
    global DATA_LEN

    train_shape = 322504
    indices = np.arange(train_shape)
    np.random.seed(5)
    np.random.shuffle(indices)
    n = len(indices)
    df = pd.read_csv('small_data.csv')

    AE_TIMESTEPS = 64
    AE_WINDOWS_STEP = 32
    AE_N_FEATURES = 1025
    AE_BATCH_SIZE = 128
    AE_EPOCHS = 1
    SMALL_DATA_INDICES = indices[:train_shape//10]
    AE_TRAIN_IDX = SMALL_DATA_INDICES[:int(n*0.7)]
    AE_VALIDATION_IDX = SMALL_DATA_INDICES[int(n*0.7):]
    AE_LATENT_DIM = 128
    DATA_LEN = (sum(df['length'])//64-31)//AE_WINDOWS_STEP
