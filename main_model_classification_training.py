
import path_configs # noqa
import tensorflow as tf
import settings
from datetime import datetime
import os
settings.init()

from modules.ClassifierGenerators import _generator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# assumes that array is not zero
def scaled(tensor):
    return (tensor-tf.math.reduce_min(tensor))/(tf.math.reduce_max(tensor)-tf.
                                                math.reduce_min(tensor))


dataset_train = tf.data.Dataset.from_generator(_generator,
                                         args=['train'],
                                         output_types=(tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32),
                                         output_shapes=((128, 64), (None), (None)))

dataset_val = tf.data.Dataset.from_generator(_generator,
                                            args=['val'],
                                            output_types=(tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32),
                                            output_shapes=((128, 64), (None), (None)))


def base_model(inputs):
    base_layer = tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                        activation='relu')(inputs)
    base_layer = tf.keras.layers.Dropout(0.4)(base_layer)
    base_layer = tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                                        activation='relu')(base_layer)
    base_layer = tf.keras.layers.Dropout(0.4)(base_layer)
    base_layer = tf.keras.layers.Flatten()(base_layer)
    return base_layer


# age branch
def build_age_branch(inputs):
    age_layer = tf.keras.layers.Dense(512, activation='relu')(inputs)
    age_layer = tf.keras.layers.Dropout(0.4)(age_layer)
    age_layer = tf.keras.layers.Dense(256, activation='relu')(age_layer)
    age_layer = tf.keras.layers.Dropout(0.4)(age_layer)
    age_output = tf.keras.layers.Dense(9, activation='sigmoid',
                                       name='age')(age_layer)
    return age_output


# gender branch,
def build_gender_branch(inputs):
    gender_layer = tf.keras.layers.Dense(512, activation='relu')(inputs)
    gender_layer = tf.keras.layers.Dropout(0.4)(gender_layer)
    gender_layer = tf.keras.layers.Dense(256, activation='relu')(gender_layer)
    gender_layer = tf.keras.layers.Dropout(0.4)(gender_layer)
    gender_output = tf.keras.layers.Dense(1, activation='sigmoid',
                                          name='gender')(gender_layer)
    return gender_output


def assemble_full_model():

    inputs = tf.keras.Input(shape=(128, 64, 1), batch_size=1)

    inputs = base_model(inputs)

    age_branch = build_age_branch(inputs)
    gender_branch = build_gender_branch(inputs)

    model = tf.keras.models.Model(inputs=inputs,
                                  outputs=[age_branch, gender_branch],
                                  name="voice_net")
    return model


model = assemble_full_model()

model.compile(
    optimizer='adam',
    loss={'age': 'sparse_categorical_crossentropy',
          'gender': 'categorical_crossentropy'},
    loss_weights={'age': 1,
                  'gender': 1},
    metrics=['accuracy'])
print(model.summary())

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M")

NAME = ("main_model_" + date_time)

for folder_name in ['logs', 'saved_models', os.path.join('saved_models',
                                                         'checkpoints')]:
    try:
        os.mkdir(folder_name)
        print("Directory", folder_name,  "created ")
    except FileExistsError:
        print("Directory", folder_name,  "already exists")


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs', str(NAME))),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('saved_models', 'checkpoints',
                              '{}.h5'.format(NAME)),
        monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    ]

tf.keras.utils.plot_model(model,
                          to_file=os.path.join('figures',
                                               'main_model_schema.pdf'),
                          show_shapes=True, expand_nested=True)

model.fit(
    dataset_train().prefetch(tf.data.experimental.AUTOTUNE),
    epochs=settings.MODEL_EPOCHS,
    steps_per_epoch=(settings.TRAIN_DATA_LEN//2)//settings.MODEL_BATCH_SIZE,
    verbose=1,
    validation_data=dataset_val()
    .prefetch(tf.data.experimental.AUTOTUNE),
    validation_steps=(settings.VAL_DATA_LEN//2)//settings.MODEL_BATCH_SIZE,
    callbacks=callbacks
    )

model.save(os.path.join('saved_models', '{}.h5'.format(NAME)))
