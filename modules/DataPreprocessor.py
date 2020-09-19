import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt


class DataLoader:
    frame_length = 2048
    frame_step = 64
    train = pd.read_csv('data_info.csv')

    def _init_(self, frame_length, frame_step):
        self.frame_length = frame_length
        self.frame_step = frame_step

    def load_audio_binary(self, case_num, train=train):
        path = str('data/en/clips/' + train['path'].loc[case_num])
        audio_binary = tf.io.read_file(path)
        audio = tfio.audio.decode_mp3(audio_binary)
        return audio

    def _tf_log10(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def power_to_db(self, S, amin=1e-16, top_db=80.0):
        ref = tf.reduce_max(S)
        log_spec = 10.0 * self._tf_log10(tf.maximum(amin, S))
        log_spec -= 10.0 * self._tf_log10(tf.maximum(amin, ref))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    def make_spectrogram(self, case_num):
        audio = self.load_audio_binary(case_num)
        audiofft = tf.signal.stft(tf.squeeze(audio), frame_length=self
                                  .frame_length, frame_step=self.frame_step)
        fftabs = tf.transpose(tf.math.abs(audiofft))
        log_fft = self.power_to_db(fftabs)
        return log_fft

    def get_labels(self, case_num, train=train):
        y_age = train['age'].map({'teens': 0, 'twenties': 1, 'seventies': 2,
                                  'fifties': 3, 'fourties': 4, 'thirties': 5,
                                  'sixties': 6, 'eighties': 7, 'nineties': 8}
                                 ).iloc[case_num]
        y_gender = train['gender'].map({'male': 0, 'female': 1}
                                       ).iloc[case_num]
        return (y_age, y_gender)

    def show_spectra(self, case_num):
        log_fft = self.make_spectrogram(case_num)
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(log_fft, cmap='plasma', interpolation='nearest',
                  aspect='auto')
        plt.gca().invert_yaxis()
        fig.show()
        # ax.set_yscale('symlog')
