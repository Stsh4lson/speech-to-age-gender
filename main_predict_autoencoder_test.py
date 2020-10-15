import path_configs # noqa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import settings
settings.init()
from modules.EncoderGenerators import TrainEncoderGenerator # noqa


# assumes that array is not zero
def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))


samples = []

for x in TrainEncoderGenerator().map(scaled, num_parallel_calls=tf.data.
                                     experimental.AUTOTUNE):
    samples.append(x)
    if len(samples) > 5:
        break

autoencoder = tf.keras.models.load_model('model_autoencoder_final.h5')
encoder = autoencoder.layers[1]
decoder = autoencoder.layers[2]
encoder = tf.keras.models.Model(encoder.layers[0].input, encoder.layers[2]
                                .output)
decoder_input = tf.keras.layers.Input(shape=(128,))
decoder = tf.keras.models.Model(decoder_input,
                                autoencoder.layers[-1](decoder_input))

yhat = encoder.predict(samples[4])
fig, ax = plt.subplots(nrows=5, figsize=(12, 6))
ax[0].plot(tf.signal.inverse_stft(tf.cast(np.vstack(samples[4]), tf.complex64), frame_length=2048, frame_step=64)[::-1])
ax[1].imshow(np.swapaxes(np.vstack(samples[4]), 0, 1), cmap='plasma', interpolation='nearest', aspect='auto')
ax[2].imshow(np.swapaxes(np.vstack(yhat), 0, 1), cmap='gray')
ax[3].imshow(np.swapaxes(np.vstack(decoder.predict(yhat)), 0, 1),
             cmap='plasma', interpolation='nearest', aspect='auto')
ax[4].plot(tf.signal.inverse_stft(tf.cast(np.vstack(decoder.predict(yhat)), tf.complex64), frame_length=2048, frame_step=64)[::-1])
fig.savefig(os.path.join('figures', 'encoded.pdf'), bbox_inches='tight')
plt.show()
