import pandas as pd
import tensorflow as tf
# import tensorflow_io as tfio
import matplotlib.pyplot as plt
import librosa
import numpy as np
from utils.data_load import data_load
import tensorflow_io as tfio

class DataPreprocessor:
    """[summary]
    """
    def __init__(self, window_time=22, frame_length=None, frame_step=None, sample_rate=48000, verbose=1):
        self.sample_rate = sample_rate
        self.window_time = window_time

        if frame_length:
            self.frame_length = frame_length
        else:
            self.frame_length = int(48000*window_time/1000)

        if frame_step:
            self.frame_step = frame_step
        else:
            self.frame_step = self.frame_length//4
            
        data_load()
        self.data_df = pd.read_csv('data_info.csv')

    def scaled_array(self, array):
        return (array - np.min(array))/(np.max(array) - np.min(array))

    def get_path(self, case_num):
        return str('data/en/clips/' + self.data_df['path'].loc[case_num])

    def normalize_audio(self, tensor):
        return ((tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.math.reduce_min(tensor))*2)-1

    def cut_voice(self, audio):
        """
        Cuts out silence from begining and end of audio sample
        and normalizes it from 1 to -1.

        Args:
            audio (ndarray): 1D audio array

        Returns:
            [ndarray]: 1D audio array
        """        
        audio = self.normalize_audio(audio)
        treshold = 0.15
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
            treshold_plot[tail] = 1
        except Exception:
            pass
        audio = audio[:treshold_x[np.where(treshold_plot == 1)[0][-1]]]
        audio = audio[treshold_x[np.where(treshold_plot == 1)[0][0]]:]
        return audio

    # def load_audio_binary(self, path):
    #     binary = pydub.AudioSegment.from_mp3(str(path))
    #     audio = np.array(binary.get_array_of_samples())
    #     return audio
    
    def load_audio_binary(self, path):
        audio_binary = tf.io.read_file(path)
        audio = tfio.audio.decode_mp3(audio_binary)
        return audio

    def make_spectrogram(self, path):
        audio = self.load_audio_binary(path)
        audio = self.cut_voice(audio)
        audio_spec = librosa.feature.melspectrogram(y=audio.numpy(),
                                                    sr=self.sample_rate,
                                                    n_fft=self.frame_length,
                                                    hop_length=self.frame_step,
                                                    n_mels=128)
        db_audio_spec = librosa.power_to_db(audio_spec,
                                            ref=1.0,
                                            top_db=80.0)
        return db_audio_spec

    def pad_spec(self, audio_mel_spec, padding_length=128):
        padding = int(np.ceil(audio_mel_spec.shape[1]/padding_length)*padding_length - audio_mel_spec
                      .shape[1])
        return np.pad(audio_mel_spec, (((0, 0), (0, padding))), mode='wrap')

    def pad_spec_3d(self, audio_mel_spec):
        padding = int(np.ceil(audio_mel_spec.shape[1]/128)*128 - audio_mel_spec
                      .shape[1])
        return np.pad(audio_mel_spec, (((0, 0), (0, padding), (0, 0))), mode='wrap')

    def show_spectra(self, case_num):
        path = self.get_path(case_num)
        db_fft = self.scaled_array(self.make_spectrogram(path))
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(db_fft, cmap='plasma', interpolation='nearest',
                  aspect='auto')
        plt.show()
        # ax.set_yscale('symlog')
        
if __name__ == "__main__":
    DP = DataPreprocessor()
    DP.show_spectra(56)