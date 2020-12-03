import time
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras import backend as K
import os

end_dense_layers = [0, 1, 2]
end_dense_sizes = [32, 64, 128]
mfcc_conv_layers = [0, 1]
mfcc_dense_layers = [0, 1]
spec_conv_layers = [1, 2, 3]

input_shape = [128, 128, 1]
input2_shape = [40, 128, 1]


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
    dset = tf.data.TFRecordDataset(path_to_file, compression_type='ZLIB')
    dset = dset.map(_parse_record, num_parallel_calls=parallel_load)
    return dset

c=0
for end_dense_layer in end_dense_layers:
    for end_dense_size in end_dense_sizes:
        for mfcc_conv_layer in mfcc_conv_layers:
            for mfcc_dense_layer in mfcc_dense_layers:
                for spec_conv_layer in spec_conv_layers:
                    NAME = "{}x{}_endDense_{}conv_{}dense_mfcc_{}spec_conv_{}".format(end_dense_layer,
                                                                                      end_dense_size,
                                                                                      mfcc_conv_layer,
                                                                                      mfcc_dense_layer,
                                                                                      spec_conv_layer,
                                                                                      int(time.time()))
                    c+=1
                    if c>36:                    
                        try:
                            """DEFINING MODEL"""
                            def ConvBlock(x):
                                x = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                                        activation='relu')(x)
                                x = tf.keras.layers.MaxPool2D()(x)
                                for l in range(spec_conv_layer-1):
                                    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                                            activation='relu')(x)
                                    x = tf.keras.layers.MaxPool2D()(x)

                                x = tf.keras.layers.Flatten()(x)
                                x = tf.keras.layers.Dropout(0.1)(x)
                                return x

                            def InputDenseBlock_mfcc(x):
                                for l in range(mfcc_dense_layer):
                                    x = tf.keras.layers.Dense(64, activation='relu')(x)
                                for l in range(mfcc_conv_layer):
                                    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                                            activation='relu')(x)
                                x = tf.keras.layers.Flatten()(x)
                                x = tf.keras.layers.Dropout(0.1)(x)
                                return x

                            def simple_model_DenseBlock_age(x):
                                for l in range(end_dense_layer):
                                    x = tf.keras.layers.Dense(
                                        end_dense_size, activation='relu')(x)
                                    x = tf.keras.layers.Dropout(0.1)(x)
                                x = tf.keras.layers.BatchNormalization()(x)
                                x = tf.keras.layers.Dense(1)(x)
                                x = tf.keras.layers.Activation(
                                    'linear', name='y_age')(x)
                                return x

                            def assemble_full_model():
                                inputs = tf.keras.Input(shape=input_shape, name='x')
                                inputs2 = tf.keras.Input(
                                    shape=input2_shape, name='x_mfcc')

                                conv_block = ConvBlock(inputs)
                                densed_mfcc_block = InputDenseBlock_mfcc(inputs2)

                                merged_inputs = tf.keras.layers.concatenate(
                                    [conv_block, densed_mfcc_block])

                                age_branch = simple_model_DenseBlock_age(merged_inputs)

                                model = tf.keras.models.Model(inputs=[inputs, inputs2],
                                                            outputs=[age_branch],
                                                            name='thragoid')
                                return model
                            model = assemble_full_model()
                            model.compile(optimizer='adam',
                                        loss='mse',
                                        metrics=['mae', 'mse'])
                            print('Model {}:'.format(c), model.count_params(), 'params')
                            """SETTING GENERATOR UP"""
                            train_set_length = 356456
                            val_set_length = 89038
                            """know lengths"""
                            batch_size = 64
                            epochs = 3
                            steps_per_epoch = train_set_length//batch_size//5
                            validation_steps = val_set_length//batch_size//2

                            x_size_dict = {'x': [128, 128, 1], 'x_mfcc': [40, 128, 1]}
                            y_size_dict = {'y_age': [1], 'y_gender': [1]}
                            dataset_train = read_TFRecord(x_size_dict, y_size_dict, 2, os.path.join(
                                'data', 'tf_record', 'data_train.tfrecord'))
                            dataset_val = read_TFRecord(x_size_dict, y_size_dict, 2, os.path.join(
                                'data', 'tf_record', 'data_val.tfrecord'))

                            def delete_y2(x, y):
                                return x, y['y_age']

                            def preprocess_dataset(dataset, batch_size):
                                dataset = dataset.map(delete_y2)
                                dataset = dataset.shuffle(
                                    buffer_size=train_set_length//50, seed=666)
                                dataset = dataset.batch(batch_size)
                                dataset = dataset.prefetch(buffer_size=2)
                                dataset = dataset.repeat(count=-1)
                                return dataset

                            dataset_train = preprocess_dataset(
                                dataset_train, batch_size)
                            dataset_val = preprocess_dataset(dataset_val, batch_size)

                            """TRAINING MODEL"""
                            log_dir = log_dir = os.path.join('logs_test', str(NAME))
                            callbacks = [tf.keras.callbacks.TensorBoard(log_dir)]

                            with tf.device('/device:GPU:0'):
                                model.fit(dataset_train,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=dataset_val,
                                        validation_steps=validation_steps,
                                        callbacks=callbacks
                                        )
                            tf.keras.backend.clear_session()
                        except Exception:
                            print('bruh')