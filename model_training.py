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


metadata = pd.read_csv(os.path.join('data', 'tf_record', 'metadata.csv'))
train_set_length = metadata[metadata.set_type == 'train'].shape[0]
val_set_length = metadata[metadata.set_type == 'val'].shape[0]
test_set_legth = metadata[metadata.set_type == 'test'].shape[0]

batch_size = 64
epochs = 50
steps_per_epoch = train_set_length//batch_size
validation_steps = val_set_length//batch_size


x_size_dict = {'x': [128, 256, 1]}
y_size_dict = {'y_age': [1], 'y_gender': [1]}

dataset_train = read_TFRecord(x_size_dict, y_size_dict, 2,
                              os.path.join('data', 'tf_record',
                                           'data_train.tfrecord'))
dataset_val = read_TFRecord(x_size_dict, y_size_dict, 2,
                            os.path.join('data', 'tf_record',
                                         'data_val.tfrecord'))


def to_rgb(x, y):
    output = tf.repeat(x, 3, axis=2)
    return output, y


def to_classes(x, y):
    y = tf.cast(y, tf.dtypes.int32)
    y = tf.one_hot(y, 8)
    y = tf.reshape(y, [8])
    y = tf.cast(y, tf.dtypes.int32)
    x = tf.cast(x, tf.dtypes.float32)
    return x, y


def random_roll(x, y):
    offset = tf.random.uniform((), minval=0, maxval=254, dtype=tf.dtypes.int32)
    x = tf.roll(x, offset, 1)
    # x = tf.image.random_brightness(x, 0.2)
    return x, y


def rescale_image(x, y):
    return (x-tf.math.reduce_min(x))/(tf.math.reduce_max(x)-tf.math.reduce_min(x)), y


def random_cut(x, y):
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
                                 tf.gather(x,
                                           tf.range(position+length, axis_len),
                                           axis=axis)), axis=axis)
        x = tf.ensure_shape(x, (128, 256, 3))
        return [x, i, amount_of_stripes]

    output = tf.while_loop(c, body, [x, i, amount_of_stripes])[0]
    return output, y


def random_stretch(x, y):
    target_width = tf.random.uniform(
        (), minval=160, maxval=256, dtype=tf.dtypes.int32)
    output = tf.image.random_crop(x, size=[128, target_width, 1])
    output = tf.image.resize(output, size=[128, 256])
    return output, y


# def flatten(x, y):
#     big = tf.concat((tf.expand_dims(tf.gather(x, 0, axis=2), 2),
#                      tf.expand_dims(tf.gather(x, 1, axis=2), 2),
#                      tf.expand_dims(tf.gather(x, 2, axis=2), 2)), axis=0)
#     return big, y


# def inputs3(x, y):
#     return {'x1': tf.expand_dims(tf.gather(x, 0, axis=2), 2),
#             'x2': tf.expand_dims(tf.gather(x, 1, axis=2), 2),
#             'x3': tf.expand_dims(tf.gather(x, 2, axis=2), 2)}, y


def preprocess_dataset(dataset, batch_size, train):
    dataset = dataset.map(lambda x, y: (x['x'], y['y_age']))
    # dataset = dataset.map(lambda x, y: {'x': x, 'predictions': y})
    dataset = dataset.map(rescale_image)

    if train:
        dataset = dataset.map(random_roll)
        dataset = dataset.map(random_stretch)
    #     dataset = dataset.map(random_cut, num_parallel_calls=12)
    #     dataset = dataset.map(rescale_image)

    dataset = dataset.map(to_classes)
    dataset = dataset.map(to_rgb)
    dataset = dataset.shuffle(
        buffer_size=train_set_length//50,
        seed=795797950,
        reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=-1)
    return dataset


dataset_train = preprocess_dataset(dataset_train, batch_size, train=True)
dataset_val = preprocess_dataset(dataset_val, batch_size, train=False)


now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M")
NAME = ("model_log" + date_time + 'MobileNetV2')
log_dir = os.path.join('logs', str(NAME))

model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 256, 3),
        classes=8,
        weights=None,
        classifier_activation="softmax"
        )


opt = tf.keras.optimizers.SGD()
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt,
              loss=loss,
              metrics='accuracy')

callbacks = [tf.keras.callbacks.TensorBoard(log_dir),
             tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('saved_models', 'checkpoints','{}.h5'.format(NAME)),
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='auto'),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              verbose=1,
                                              patience=10)
             ]
model.fit(
    dataset_train,
    steps_per_epoch=steps_per_epoch//4,
    epochs=50,
    verbose=1,
    validation_data=dataset_val,
    validation_steps=validation_steps,
    callbacks=callbacks
    )
