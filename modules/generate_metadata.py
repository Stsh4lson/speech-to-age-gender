import tensorflow as tf         # tensorflow for deep learning
import os                       # os interaction
import pandas as pd             # data manipulation
from tqdm import tqdm


def generate_metadata(path):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def read_TFRecord(input_shape: dict,
                      output_shape: dict,
                      parallel_load: int,
                      path_to_file: str):
        """
        Method for reading .tfrecord file into TFRecordDataset object.
        :param input_shape: input_shape of dataset
        :param output_shape: output_shape of dataset
        :param batch_size: batch size
        :param parallel_load: number of parallel loading threads
        :return: TFRecordDataset object
        """
        featdef = dict()
        for key in input_shape.keys():
            featdef[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        for key in output_shape.keys():
            featdef[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

        def _parse_record(example_proto):
            """Parse a single record into image, weather labels, ground labels"""
            example = tf.io.parse_single_example(example_proto, featdef)
            x_dict = dict()
            for x_key in [k for k in example.keys() if 'x' in k]:
                data = tf.io.decode_raw(example[x_key], tf.float32)
                x_dict[x_key] = tf.reshape(data, input_shape[x_key])
            y_dict = dict()
            for y_key in [k for k in example.keys() if 'y' in k]:
                data = tf.io.decode_raw(example[y_key], tf.float32)
                y_dict[y_key] = tf.reshape(data, output_shape[y_key])
            return x_dict, y_dict
        dset = tf.data.TFRecordDataset(path_to_file, compression_type="ZLIB")
        dset = dset.map(_parse_record, num_parallel_calls=parallel_load)
        return dset

    x_size_dict = {'x': [128, 256, 1]}
    y_size_dict = {'y_age': [1], 'y_gender': [1]}

    dataset_train = read_TFRecord(x_size_dict, y_size_dict, 2,
                                  os.path.join('data', 'tf_record',
                                               'data_train.tfrecord'))
    dataset_val = read_TFRecord(x_size_dict, y_size_dict, 2,
                                os.path.join('data', 'tf_record',
                                             'data_val.tfrecord'))
    dataset_test = read_TFRecord(x_size_dict, y_size_dict, 2,
                                 os.path.join('data', 'tf_record',
                                              'data_test.tfrecord'))

    def dataset_labels(dataset):
        label_age = []
        label_gender = []
        for x, y in tqdm(dataset):
            label_age.append(y['y_age'].numpy()[0])
            label_gender.append(y['y_gender'].numpy()[0])
        return label_age, label_gender

    train_labels = dataset_labels(dataset_train)
    val_labels = dataset_labels(dataset_val)
    test_labels = dataset_labels(dataset_test)

    train_labels_df = pd.DataFrame(train_labels).transpose()
    val_labels_df = pd.DataFrame(val_labels).transpose()
    test_labels_df = pd.DataFrame(test_labels).transpose()

    train_labels_df['set_type'] = 'train'
    val_labels_df['set_type'] = 'val'
    test_labels_df['set_type'] = 'test'

    meta_df = pd.concat([train_labels_df,
                         val_labels_df,
                         test_labels_df
                         ])

    filepath = os.path.join(path, 'metadata.csv')
    meta_df.to_csv(filepath, index=False)
    print('metadata saved to', filepath)
