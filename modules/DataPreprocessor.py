import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import librosa
import numpy as np


class DataLoader:
    sample_rate = 48000
    window_time = 22
    frame_length = int(48000*window_time/1000)
    frame_step = frame_length//4
    train = pd.read_csv('data_info.csv')

    def _init_(self, window_time, frame_length, frame_step, sample_rate):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.window_time = window_time

    def scaled_array(self, array):
        return (array - np.min(array))/(np.max(array) - np.min(array))

    def get_path(self, case_num, train=train):
        return str('data/en/clips/' + train['path'].loc[case_num])

    def cut_voice(self, audio):
        treshold = 0.1
        treshold_plot = []
        treshold_x = []

        window = int(48000*0.02)
        window_start = 0
        window_end = window
        L = len(audio)
        while window_start < L:
            limit = min(window_end, L)
            if np.max(audio[window_start:limit]) >= treshold:
                treshold_plot.append(True)
            else:
                treshold_plot.append(False)
            treshold_x.append(window_start)
            window_start += window
            window_end += window
        
        treshold_plot = np.array(treshold_plot)
        tail = np.arange(np.where(treshold_plot == 1)[0][-1],
                         np.where(treshold_plot == 1)[0][-1]+10, 1)
        try:
            treshold_plot[tail]=1
        except Exception:
            print('bruh')
        audio = audio[:treshold_x[np.where(treshold_plot == 1)[0][-1]]]
        audio = audio[treshold_x[np.where(treshold_plot == 1)[0][0]]:]
        return audio

    def load_audio_binary(self, case_num):
        path = self.get_path(case_num)
        audio_binary = tf.io.read_file(path)
        audio = tfio.audio.decode_mp3(audio_binary)
        return audio[:, 0]

    def make_spectrogram(self, case_num):
        audio = self.load_audio_binary(case_num).numpy()
        audio = self.cut_voice(audio)
        audio_spec = librosa.feature.melspectrogram(audio,
                                                    sr=self.sample_rate,
                                                    n_fft=self.frame_length,
                                                    hop_length=self.frame_step,
                                                    n_mels=128)
        db_audio_spec = librosa.power_to_db(audio_spec,
                                            ref=1.0,
                                            top_db=80.0)
        return db_audio_spec

    def pad_spec(self, audio_mel_spec):
        padding = int(np.ceil(audio_mel_spec.shape[1]/128)*128 - audio_mel_spec
                      .shape[1])
        return np.pad(audio_mel_spec, (((0, 0), (0, padding))), mode='wrap')

    def get_labels(self, case_num, train=train):
        y_age = train['age'].map({'teens': 0, 'twenties': 1, 'seventies': 2,
                                  'fifties': 3, 'fourties': 4, 'thirties': 5,
                                  'sixties': 6, 'eighties': 7, 'nineties': 8}
                                 ).iloc[case_num]
        y_gender = train['gender'].map({'male': 0, 'female': 1}
                                       ).iloc[case_num]
        return (y_age, y_gender)

    def show_spectra(self, case_num):
        db_fft = self.scaled_array(self.make_spectrogram(case_num))
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(db_fft, cmap='plasma', interpolation='nearest',
                  aspect='auto')
        fig.show()
        # ax.set_yscale('symlog')
