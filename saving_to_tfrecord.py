import tensorflow as tf
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
from modules.data_preprocessor import DataPreprocessor
from modules.generate_metadata import generate_metadata

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def append_to_TFRecord(writer: tf.io.TFRecordWriter, x_dict: dict,
                       y_dict: dict) -> None:
    """
    Append data to open writer.
    :param writer: TFRecordWriter
    :param x_dict: dict with np.arrays
    :param y_dict: dict with np.arrays
    :return: None
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    features = dict()
    for key in x_dict.keys():
        features[key] = _bytes_feature(tf.compat.as_bytes(
            x_dict[key].astype(np.float32).tostring()))
    for key in y_dict.keys():
        features[key] = _bytes_feature(tf.compat.as_bytes(
            y_dict[key].astype(np.float32).tostring()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())


def scale(array):
    return (array-np.min(array))/(np.max(array)-np.min(array))


dp = DataPreprocessor()
path = os.path.join('data', 'tf_record')
Path(path).mkdir(parents=True, exist_ok=True)
options = tf.io.TFRecordOptions(compression_level=1, compression_type="ZLIB")

for set_type in dp.train.set_type.unique():
    # setting set_type for tf.record file
    print('\nSaving', str(set_type), 'set to:', os.path.join(
        path, ('data_' + str(set_type) + '.tfrecord')))
    train_df = dp.train[dp.train.set_type == set_type]
    # mapping string labels to floats
    train_df.age = train_df.age.map({'teens': 0, 'twenties': 1,
                                     'seventies': 6, 'fifties': 4,
                                     'fourties': 3, 'thirties': 2,
                                     'sixties': 5, 'eighties': 7}
                                    )
    train_df.gender = train_df.gender.map({'male': 0, 'female': 1})

    file_path = os.path.join(path, ('data_' + str(set_type) + '.tfrecord'))
    with tf.io.TFRecordWriter(file_path, options=options) as writer:
        for spec_num in tqdm(train_df.index.to_numpy()):
            # getting spectrograms and labels from data_loader
            # audio = np.array(dp.cut_voice(dp.load_audio_binary(spec_num)))
            # mfcc = librosa.feature.mfcc(audio,
            #                             n_mfcc=128,
            #                             sr=dp.sample_rate,
            #                             n_fft=dp.frame_length,
            #                             hop_length=dp.frame_step,
            #                             fmin=20, fmax=8000, lifter=20)
            # try:
            #     mfcc_delta = librosa.feature.delta(mfcc, width=5, axis=1)
            # except Exception:
            #     # because when mode='interp', width = 5
            #     # cannot exceed data.shape[axis]=4
            #     mfcc_delta = mfcc

            spec = dp.make_spectrogram(spec_num)
            # spec_3d = np.concatenate([scale(np.expand_dims(spec, 2)),
            #                           scale(np.expand_dims(mfcc, 2)),
            #                           scale(np.expand_dims(mfcc_delta, 2))],
            #                          axis=2)
            paded_spec = dp.pad_spec(spec)
            age = train_df.loc[spec_num][2]
            gender = train_df.loc[spec_num][3]
            # windowing spectrograms
            for window in np.arange(0, paded_spec.shape[1]-128, 128):
                spec_window = paded_spec[:, window:window + 256]
                # making dicts for tfrecord writer
                x_dict = {'x': np.array(spec_window)}
                y_dict = {'y_age': np.array(age), 'y_gender': np.array(gender)}
                append_to_TFRecord(writer=writer, x_dict=x_dict, y_dict=y_dict)


generate_metadata(path)
