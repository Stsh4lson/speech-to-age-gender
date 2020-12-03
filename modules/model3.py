import tensorflow as tf

class Model:
    input_shape = [128, 128, 1]
    input2_shape = [40, 128, 1]

    def _init_(self, input_shape, input2_shape):
        self.input_shape = input_shape
        self.input2_shape = input2_shape

    def simple_model_ConvBlock(self, inputs):
        base_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                            activation='relu')(inputs)
        base_layer = tf.keras.layers.MaxPool2D()(base_layer)
        base_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                            activation='relu')(base_layer)
        base_layer = tf.keras.layers.Flatten()(base_layer)
        base_layer = tf.keras.layers.Dropout(0.2)(base_layer)
        return base_layer

    def InputDenseBlock_mfcc(self, x):
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    # def simple_model_DenseBlock_gender(self, conv_outputs):
    #     X = tf.keras.layers.Dense(256, activation='relu')(conv_outputs)
    #     X = tf.keras.layers.Dropout(0.1)(X)
    #     X = tf.keras.layers.Dense(128, activation='relu')(X)
    #     X = tf.keras.layers.Dropout(0.1)(X)
    #     X = tf.keras.layers.BatchNormalization()(X)
    #     X = tf.keras.layers.Dense(1)(X)
    #     gender_output = tf.keras.layers.Activation('softmax', name='y_gender')(X)
    #     return gender_output

    def simple_model_DenseBlock_age(self, x):
        X = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)
        X = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False)(X)
        X = tf.keras.layers.Dense(64)(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(1)(X)
        age_output = tf.keras.layers.Activation('linear', name='y_age')(X)
        return age_output

    def assemble_full_model(self): 
        inputs = tf.keras.Input(shape=self.input_shape, name='x')
        inputs2 = tf.keras.Input(shape=self.input2_shape, name='x_mfcc')

        conv_block = self.simple_model_ConvBlock(inputs)
        densed_mfcc_block  = self.InputDenseBlock_mfcc(inputs2)

        merged_inputs = tf.keras.layers.concatenate(inputs=[conv_block, densed_mfcc_block], axis=1)

        reshaped = tf.keras.layers.Reshape((168, 128*64), input_shape=(168, 128, 64))(merged_inputs)
        age_branch = self.simple_model_DenseBlock_age(reshaped)
        # gender_branch = self.simple_model_DenseBlock_gender(merged_inputs)

        model = tf.keras.models.Model(inputs=[inputs, inputs2],
                                    outputs=[age_branch],
                                    name='thragoid')
        return model
