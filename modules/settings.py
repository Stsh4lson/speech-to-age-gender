import numpy as np

def init():
    global AE_TIMESTEPS
    global AE_WINDOWS_STEP
    global AE_N_FEATURES
    global AE_BATCH_SIZE
    global AE_EPOCHS
    global AE_TRAIN_IDX
    global AE_VALIDATION_IDX
    global AE_LATENT_DIM

    AE_TIMESTEPS = 32
    AE_WINDOWS_STEP = 16
    AE_N_FEATURES = 1025
    AE_BATCH_SIZE = 128
    AE_EPOCHS = 50
    AE_TRAIN_IDX = np.arange(0, 225752, 1)
    AE_VALIDATION_IDX = np.arange(225752, 322504, 1)
    AE_LATENT_DIM = 128