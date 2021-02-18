import tensorflow as tf
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import csv
from utils.data_preprocessor import DataPreprocessor
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class SavingToTFRecord():
    """
    Reads paths and labels from data_info.csv and saves to data/tf_records.
    Also logs metadata about binary files into csv files for verification.
    """

    def __init__(self) -> None:
        self.data_save_path = os.path.join('data', 'tf_record')
        Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
        self.options = tf.io.TFRecordOptions(compression_level=1,
                                             compression_type="ZLIB")

    def append_to_TFRecord(self, writer: tf.io.TFRecordWriter, x_dict: dict,
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

    def scale(self, array) -> np.ndarray:
        return (array-np.min(array))/(np.max(array)-np.min(array))

    def generate_files(self, window_size=256, unique=False) -> None:
        """Generates mel spectrogram, cuts into windows and saves to tfrecord files

        Args:
            window_size (int): Width of the window cut from the calculated spectrogram
            unique (bool): If True, cut sample and save only first window from that sample
        """
        dp = DataPreprocessor()
        for set_type in dp.train.set_type.unique():
            # setting set_type for tf.record file
            print('\nSaving', str(set_type), 'set to:', os.path.join(
                self.data_save_path, ('data_' + str(set_type) + '.tfrecord')))
            train_df = dp.train[dp.train.set_type == set_type]
            # mapping string labels to floats
            train_df.age = train_df.age.map({'teens': 0, 'twenties': 1,
                                            'seventies': 6, 'fifties': 4,
                                            'fourties': 3, 'thirties': 2,
                                            'sixties': 5, 'eighties': 7}
                                            )
            train_df.gender = train_df.gender.map({'male': 0, 'female': 1})

            file_path = os.path.join(self.data_save_path,
                                     ('data_' + str(set_type) + '.tfrecord')
                                     )
            #
            metadata_path = os.path.join(self.data_save_path,
                                         f'metadata_{set_type}.csv')
            metadata_csv = open(metadata_path, 'w', newline="")
            csv_writer = csv.writer(metadata_csv)
            csv_writer.writerow(['path', 'age', 'gender'])

            with tf.io.TFRecordWriter(file_path, options=self.options) as writer:
                for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
                    audio_path = os.path.join('data', 'en', 'clips', row.path)
                    spec = dp.make_spectrogram(audio_path)
                    paded_spec = dp.pad_spec(spec, window_size//2)
                    age = row.age
                    gender = row.gender

                    # windowing spectrograms
                    for window in np.arange(0, paded_spec.shape[1]-window_size//2, window_size//2):
                        spec_window = paded_spec[:, window:window + window_size]
                        # making dicts for tfrecord writer
                        x_dict = {'x': np.array(spec_window)}
                        y_dict = {'y_age': np.array(age),
                                  'y_gender': np.array(gender)}
                        self.append_to_TFRecord(writer=writer,
                                                x_dict=x_dict,
                                                y_dict=y_dict)
                        csv_writer.writerow([row.path, age, gender])
                        metadata_csv.flush()
                        if unique:
                            break
                        else:
                            continue
            metadata_csv.close()


if __name__ == '__main__':
    save_obj = SavingToTFRecord()
    save_obj.generate_files(window_size=256, unique=False)
