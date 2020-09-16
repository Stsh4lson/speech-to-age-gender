import path_configs # noqa
import settings
import os
import time
import tensorflow as tf
settings.init()
from EncoderGenerators import (TrainEncoderGenerator, # noqa
                               ValidationEncoderGenerator)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# assumes that array is not zero
def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))


def doubleOutput(input):
    return input, input


input = tf.keras.layers.Input(shape=(settings.AE_TIMESTEPS, settings.
                                     AE_N_FEATURES))

encoded = tf.keras.layers.LSTM(settings.AE_N_FEATURES//4, activation='tanh',
                               return_sequences=True)(input)
encoded = tf.keras.layers.LSTM(settings.AE_LATENT_DIM, activation='tanh'
                               )(encoded)
decoded = tf.keras.layers.RepeatVector(settings.AE_TIMESTEPS
                                       )(encoded)
decoded = tf.keras.layers.LSTM(settings.AE_N_FEATURES//4, activation='tanh',
                               return_sequences=True)(decoded)
decoded = tf.keras.layers.LSTM(settings.AE_N_FEATURES,
                               return_sequences=True)(decoded)

autoencoder = tf.keras.models.Model(inputs=input, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mse')

NAME = "model_1_{}".format(int(time.time()))
for folder_name in ['logs', 'saved_models', 'saved_models\\checkpoints']:
    try:
        os.mkdir(folder_name)
        print("Directory", folder_name,  "created ")
    except FileExistsError:
        print("Directory", folder_name,  "already exists")

callbacks = [
    # tf.keras.callbacks.TensorBoard(log_dir='logs\\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='saved_models\\checkpoints\\{}.h5'.format(NAME),
        monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    ]

with tf.device("GPU:0"):
    autoencoder.fit(
        TrainEncoderGenerator()
        .prefetch(tf.data.experimental.AUTOTUNE)
        .map(scaled, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(doubleOutput, num_parallel_calls=tf.data.experimental.AUTOTUNE),
        epochs=settings.AE_EPOCHS,
        steps_per_epoch=len(settings.AE_TRAIN_IDX) // settings.AE_BATCH_SIZE,
        verbose=1,
        validation_data=ValidationEncoderGenerator()
        .prefetch(tf.data.experimental.AUTOTUNE)
        .map(scaled, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(doubleOutput, num_parallel_calls=tf.data.experimental.AUTOTUNE),
        validation_steps=len(settings.AE_VALIDATION_IDX) // settings
        .AE_BATCH_SIZE,
        callbacks=callbacks
        )

autoencoder.save('saved_models\\model{}_{}.h5'.format(NAME, time.time()))
print('saved')
