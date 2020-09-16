import path_configs # noqa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import settings
settings.init()
from EncoderGenerators import TESTEncoderGenerator # noqa


# assumes that array is not zero
def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))


samples = []
for x in TESTEncoderGenerator().map(scaled, num_parallel_calls=tf.data.
                                    experimental.AUTOTUNE):
    samples.append(x)
    if len(samples) > 5:
        break

autoencoder = tf.keras.models.load_model('saved_models\\latent128.h5')

yhat = autoencoder.predict(samples[3], verbose=0)
fig, ax = plt.subplots(nrows=2, figsize=(24, 12))
ax[0].imshow(np.swapaxes(np.vstack(samples[3]), 0, 1))
ax[1].imshow(np.swapaxes(np.vstack(yhat), 0, 1))
plt.show()
