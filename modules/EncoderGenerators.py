import tensorflow as tf
import numpy as np
import settings

class TrainEncoderGenerator(tf.data.Dataset):    
    def _generator(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE):          
        from DataPreprocessor import DataLoader
        dl = DataLoader()
        spectral_frames = []
        for case_num in case_nums:
            X_array = dl.make_spectrogram(case_num) #to map
            silence = np.argwhere(np.all(X_array[..., :] == -80, axis=0))
            X_array = tf.cast(np.swapaxes(np.delete(X_array, silence, axis=1), 0, 1), tf.float32)            
            L = X_array.shape[0]
            batch_start = 0
            batch_end = TIMESTEPS
            while batch_start < L-TIMESTEPS:
                limit = min(batch_end, L)
                spectral_frame = X_array[batch_start:limit, :]
                spectral_frames.append(spectral_frame)
                if len(spectral_frames) >= BATCH_SIZE:
                    spectral_frames = tf.stack(spectral_frames)
                    yield(spectral_frames)
                    spectral_frames = []
                batch_start += WINDOWS_STEP #window sliding by half of a window size
                batch_end += WINDOWS_STEP
    
    def __new__(cls, case_nums=settings.AE_TRAIN_IDX, TIMESTEPS=settings.AE_TIMESTEPS,
    WINDOWS_STEP=settings.AE_WINDOWS_STEP, N_FEATURES=settings.AE_N_FEATURES, BATCH_SIZE=settings.AE_BATCH_SIZE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float32,
            output_shapes=(BATCH_SIZE, TIMESTEPS, N_FEATURES),
            args=(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE)
        )

class ValidationEncoderGenerator(tf.data.Dataset):
    def _generator(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE):        
        from DataPreprocessor import DataLoader
        dl = DataLoader()
        spectral_frames = []
        for case_num in case_nums:
            X_array = dl.make_spectrogram(case_num) #to map
            silence = np.argwhere(np.all(X_array[..., :] == -80, axis=0))
            X_array = tf.cast(np.swapaxes(np.delete(X_array, silence, axis=1), 0, 1), tf.float32)            
            L = X_array.shape[0]
            batch_start = 0
            batch_end = TIMESTEPS
            while batch_start < L-TIMESTEPS:
                limit = min(batch_end, L)
                spectral_frame = X_array[batch_start:limit, :]
                spectral_frames.append(spectral_frame)
                if len(spectral_frames) >= BATCH_SIZE:
                    spectral_frames = tf.stack(spectral_frames)
                    yield(spectral_frames)
                    spectral_frames = []
                batch_start += WINDOWS_STEP #window sliding by half of a window size
                batch_end += WINDOWS_STEP
    
    def __new__(cls, case_nums=settings.AE_VALIDATION_IDX, TIMESTEPS=settings.AE_TIMESTEPS,
    WINDOWS_STEP=settings.AE_WINDOWS_STEP, N_FEATURES=settings.AE_N_FEATURES, BATCH_SIZE=settings.AE_BATCH_SIZE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float32,
            output_shapes=(BATCH_SIZE, TIMESTEPS, N_FEATURES),
            args=(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE)
        )


class TESTEncoderGenerator(tf.data.Dataset):
    def _generator(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE):        
        from DataPreprocessor import DataLoader
        dl = DataLoader()
        spectral_frames = []
        for case_num in case_nums:
            X_array = dl.make_spectrogram(case_num) #to map
            silence = np.argwhere(np.all(X_array[..., :] == -80, axis=0))
            X_array = tf.cast(np.swapaxes(np.delete(X_array, silence, axis=1), 0, 1), tf.float32)            
            L = X_array.shape[0]
            batch_start = 0
            batch_end = TIMESTEPS
            while batch_start < L-TIMESTEPS:
                limit = min(batch_end, L)
                spectral_frame = X_array[batch_start:limit, :]
                spectral_frames.append(spectral_frame)
                if len(spectral_frames) >= BATCH_SIZE:
                    spectral_frames = tf.stack(spectral_frames)
                    yield(spectral_frames)
                    spectral_frames = []
                batch_start += WINDOWS_STEP #window sliding by half of a window size
                batch_end += WINDOWS_STEP
    def __new__(cls, case_nums=settings.AE_VALIDATION_IDX, TIMESTEPS=settings.AE_TIMESTEPS, 
    WINDOWS_STEP=settings.AE_WINDOWS_STEP, N_FEATURES=settings.AE_N_FEATURES, BATCH_SIZE=settings.AE_BATCH_SIZE):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float32,
            output_shapes=(BATCH_SIZE, TIMESTEPS, N_FEATURES),
            args=(case_nums, TIMESTEPS, WINDOWS_STEP, BATCH_SIZE)
        )