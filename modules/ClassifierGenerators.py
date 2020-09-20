import tensorflow as tf
import numpy as np
import settings

def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))

class TrainClassifierGenerator(tf.data.Dataset):
    def _generator(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE):
        autoencoder = tf.keras.models.load_model('model_autoencoder_final.h5')
        encoder = autoencoder.layers[1]

        encoder = tf.keras.models.Model(encoder.layers[0].input,
                                        encoder.layers[2].output)

        from DataPreprocessor import DataLoader
        dl = DataLoader()
        data_len = 0
        max_batch_len = BATCH_SIZE*TIMESTEPS
        audio_batch = []
        labels = []
        while True:
            for case_num in case_nums:
                X_sample = dl.make_spectrogram(case_num)
                y_age_in, y_gender_in = dl.get_labels(case_num)
                data_len += X_sample.shape[1]
                if data_len > max_batch_len:
                    X_array = np.array(np.column_stack(audio_batch))
                    X_array = tf.cast(np.swapaxes(X_array, 0, 1), tf.float32)
                    # padding to smallest n*timestep above max length
                    padding = int(max_batch_len - X_array.shape[0])
                    # padding elements to set length
                    X_array = tf.cast(X_array, tf.float32)
                    labels = tf.cast(labels, tf.float32)
                    X_array = tf.pad(X_array, ([[0, padding], [0, 0]]))
                    labels = tf.pad(labels, ([[0, padding], [0, 0]]))
                    # reshaping tensors to wanted shapes
                    X_array = tf.reshape(X_array, [1024, X_array.shape[0]//1024, 1025])
                    labels = labels[::64]

                    y_age = labels[:, 0]
                    y_gender = labels[:, 1]
                    # feature extraction from X tensor using pretrained encoder
                    latent_spectral_frames = encoder.predict(X_array)
                    yield(latent_spectral_frames, y_age, y_gender)
                    data_len = X_sample.shape[1]
                    labels = []
                    audio_batch = []

                for x in np.arange(X_sample.shape[1]):
                    labels.append((y_age_in, y_gender_in))
                audio_batch.append(scaled(X_sample))

    def __new__(cls,
                case_nums=settings.AE_TRAIN_IDX[40:],
                TIMESTEPS=settings.AE_TIMESTEPS,
                WINDOWS_STEP=settings.AE_WINDOWS_STEP,
                N_FEATURES=settings.AE_N_FEATURES,
                BATCH_SIZE=settings.MODEL_BATCH_SIZE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32),

            output_shapes=((BATCH_SIZE, 128),
                           (BATCH_SIZE, ),
                           (BATCH_SIZE, )),
            args=(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE)
        )


class ValidationClassifierGenerator(tf.data.Dataset):
    def _generator(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE):
        autoencoder = tf.keras.models.load_model('model_autoencoder_final.h5')
        encoder = autoencoder.layers[1]

        encoder = tf.keras.models.Model(encoder.layers[0].input,
                                        encoder.layers[2].output)

        from DataPreprocessor import DataLoader
        dl = DataLoader()
        data_len = 0
        max_batch_len = BATCH_SIZE*TIMESTEPS
        audio_batch = []
        labels = []
        while True:
            for case_num in case_nums:
                X_sample = dl.make_spectrogram(case_num)
                y_age_in, y_gender_in = dl.get_labels(case_num)
                data_len += X_sample.shape[1]
                if data_len > max_batch_len:
                    X_array = np.array(np.column_stack(audio_batch))
                    X_array = tf.cast(np.swapaxes(X_array, 0, 1), tf.float32)
                    # padding to smallest n*timestep above max length
                    padding = int(max_batch_len - X_array.shape[0])
                    # padding elements to set length
                    X_array = tf.cast(X_array, tf.float32)
                    labels = tf.cast(labels, tf.float32)
                    X_array = tf.pad(X_array, ([[0, padding], [0, 0]]))
                    labels = tf.pad(labels, ([[0, padding], [0, 0]]))
                    # reshaping tensors to wanted shapes
                    X_array = tf.reshape(X_array, [1024, X_array.shape[0]//1024, 1025])
                    labels = labels[::64]

                    y_age = labels[:, 0]
                    y_gender = labels[:, 1]
                    # feature extraction from X tensor using pretrained encoder
                    latent_spectral_frames = encoder.predict(X_array)
                    yield(latent_spectral_frames, y_age, y_gender)
                    data_len = X_sample.shape[1]
                    labels = []
                    audio_batch = []

                for x in np.arange(X_sample.shape[1]):
                    labels.append((y_age_in, y_gender_in))
                audio_batch.append(scaled(X_sample))

    def __new__(cls,
                case_nums=settings.AE_VALIDATION_IDX,
                TIMESTEPS=settings.AE_TIMESTEPS,
                WINDOWS_STEP=settings.AE_WINDOWS_STEP,
                N_FEATURES=settings.AE_N_FEATURES,
                BATCH_SIZE=settings.MODEL_BATCH_SIZE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32),
            output_shapes=((BATCH_SIZE, 128),
                           (BATCH_SIZE, ),
                           (BATCH_SIZE, )),
            args=(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE)
        )
