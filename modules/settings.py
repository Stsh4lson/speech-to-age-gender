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
    global DATA_INDICES
    global TRAIN_DATA_LEN
    global VAL_DATA_LEN
    global LENGTHS
    global MODEL_BATCH_SIZE
    global MODEL_EPOCHS
    global TRAIN_DATA_LEN_FULL
    global VAL_DATA_LEN_FULL
    df = pd.read_csv('data_info.csv')

    LENGTHS = df['length']
    AE_TIMESTEPS = 64
    AE_WINDOWS_STEP = 32
    AE_N_FEATURES = 1025
    AE_BATCH_SIZE = 128
    MODEL_BATCH_SIZE = 256
    MODEL_EPOCHS = 6

    AE_EPOCHS = 20
    DATA_INDICES = df.index.values
    AE_TRAIN_IDX = DATA_INDICES[:int(len(DATA_INDICES)*0.7)]
    AE_VALIDATION_IDX = DATA_INDICES[int(len(DATA_INDICES)*0.7):]
    AE_LATENT_DIM = 128
    TRAIN_DATA_LEN_FULL = sum(df.loc[AE_TRAIN_IDX]['length'])
    VAL_DATA_LEN_FULL = sum(df.loc[AE_VALIDATION_IDX]['length'])
    TRAIN_DATA_LEN = (sum(df.loc[AE_TRAIN_IDX]['length'])//64-31
                      )//AE_WINDOWS_STEP
    VAL_DATA_LEN = (sum(df.loc[AE_VALIDATION_IDX]['length'])//64-31
                    )//AE_WINDOWS_STEP
    print('TRAIN_DATA_LEN:', TRAIN_DATA_LEN)
    print('VAL_DATA_LEN:', VAL_DATA_LEN)
