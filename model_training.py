import tensorflow as tf         # tensorflow for deep learning
from datetime import datetime   # time keeping
import os                       # os interaction
import pandas as pd             # data manipulation


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.executing_eagerly())

print("\nTensorflow report:")
print("Devices:", tf.config.list_physical_devices('GPU'))
print("Version:", tf.__version__)


class Model:

    def __init__(self, epochs=50, batch_size=32):
        self.image_height = 128
        self.image_width = 1024
        metadata_train = pd.read_csv(os.path.join('data',
                                                  'tf_record',
                                                  'metadata_train.csv'))
        metadata_val = pd.read_csv(os.path.join('data',
                                                'tf_record',
                                                'metadata_val.csv'))
        self.train_set_length = metadata_train.shape[0]
        self.val_set_length = metadata_val.shape[0]
        
        self.batch_size = batch_size
        self.epochs = epochs

    def read_TFRecord(self, input_shape: dict,
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

    def to_rgb(self, x, y):
        output = tf.repeat(x, 3, axis=2)
        return output, y

    def to_classes(self, x, y):
        y = tf.cast(y, tf.dtypes.int32)
        x = tf.cast(x, tf.dtypes.float32)
        y = tf.one_hot(y, 8)
        y = tf.reshape(y, [8])
        return x, y

    def random_roll(self, x, y):
        offset = tf.random.uniform((), minval=0, maxval=self.image_width-1, dtype=tf.dtypes.int32)
        x = tf.roll(x, offset, 1)
        # x = tf.image.random_brightness(x, 0.2)
        return x, y

    def rescale_image(self, x, y):
        return (x-tf.math.reduce_min(x))/(tf.math.reduce_max(x)-tf.math.reduce_min(x)), y

    def random_cut(self, x, y):
        amount_of_stripes = tf.random.uniform(
            (), minval=0, maxval=4, dtype=tf.dtypes.int32)
        i = tf.constant(0)

        def c(x, i, amount_of_stripes):
            return tf.less_equal(i, amount_of_stripes)

        def body(x, i, amount_of_stripes):
            i = tf.add(i, 1)
            length = tf.random.uniform(
                (), minval=2, maxval=15, dtype=tf.dtypes.int32)
            axis = tf.random.uniform((), minval=0, maxval=2, dtype=tf.dtypes.int32)
            axis_len = tf.gather(x.shape, axis, axis=0)
            position = tf.random.uniform(
                (), minval=0, maxval=axis_len-length, dtype=tf.dtypes.int32)
            zeros = tf.zeros_like(x)

            x = tf.concat((tf.gather(x, tf.range(0, position), axis=axis),
                          tf.gather(zeros,
                                    tf.range(position, position+length), axis=axis),
                                             tf.gather(x, tf.range(position+length, axis_len),
                                                       axis=axis)), axis=axis)
            x = tf.ensure_shape(x, (self.image_height, self.image_width, 3))
            return [x, i, amount_of_stripes]

        output = tf.while_loop(c, body, [x, i, amount_of_stripes])[0]
        return output, y

    def random_stretch(self, x, y):
        target_width = tf.random.uniform(
            (), minval=160, maxval=self.image_width, dtype=tf.dtypes.int32)
        output = tf.image.random_crop(x, size=[self.image_height, target_width, 1])
        output = tf.image.resize(output, size=[self.image_height, self.image_width])
        return output, y

    def preprocess_dataset(self, dataset, batch_size, train):
        dataset = dataset.map(lambda x, y: (x['x'], y['y_age']))
        # dataset = dataset.map(lambda x, y: {'x': x, 'predictions': y})
        # dataset = dataset.map(self.rescale_image)

        if train:
            dataset = dataset.map(self.random_roll)
            dataset = dataset.map(self.random_stretch)            
            dataset = dataset.shuffle(
                buffer_size=self.train_set_length//20,
                seed=795797950,
                reshuffle_each_iteration=True)

        dataset = dataset.map(self.to_classes)
        dataset = dataset.map(self.to_rgb)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(count=-1)
        return dataset

    def create_dataset(self, set_type='train', preprocess=True):
        x_size_dict = {'x': [self.image_height, self.image_width, 1]}
        y_size_dict = {'y_age': [1], 'y_gender': [1]}

        if set_type == 'train':
            dataset = self.read_TFRecord(x_size_dict, y_size_dict, 2,
                                         os.path.join('data',
                                                      'tf_record',
                                                      'data_train.tfrecord'))
            if preprocess:
                dataset = self.preprocess_dataset(dataset,
                                                  self.batch_size,
                                                  train=True)
            return dataset

        elif set_type == 'val':
            dataset = self.read_TFRecord(x_size_dict, y_size_dict, 2,
                                         os.path.join('data',
                                                      'tf_record',
                                                      'data_val.tfrecord'))
            if preprocess:
                dataset = self.preprocess_dataset(dataset,
                                                  self.batch_size,
                                                  train=False)
            return dataset

        else:
            raise ValueError('Wrong set_type, use "train" or "val"')

    def train(self, logging=True):
        steps_per_epoch = self.train_set_length//self.batch_size
        validation_steps = self.val_set_length//self.batch_size

        dataset_train = self.create_dataset(set_type='train')
        dataset_val = self.create_dataset(set_type='val')

        now = datetime.now()
        date_time = now.strftime("%m_%d_%H_%M")
        NAME = ("model_log" + date_time + 'MobileNetV2')
        log_dir = os.path.join('logs', str(NAME))

        model = tf.keras.applications.MobileNetV2(
                input_shape=(self.image_height, self.image_width, 3),
                classes=8,
                weights=None,
                classifier_activation="softmax"
                )

        opt = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics='accuracy')

        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('saved_models', 'checkpoints', f'{NAME}.h5'),
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='auto'),
                     tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      verbose=1,
                                                      patience=10)
                    ]
        if logging:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir))

        model.fit(
            dataset_train,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=validation_steps,
            callbacks=callbacks
            )


if __name__ == '__main__':
    m = Model(epochs=50, batch_size=32)
    m.train(logging=False)
