import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, regularizers


def model1(input_shape, num_classes):
    input = layers.Input(shape = input_shape)
    x = layers.BatchNormalization()(input)
    x = layers.Conv2D(32, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    #x = layers.Conv2D(32, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    #x = layers.Conv2D(64, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(128, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    #x = layers.Conv2D(128, (3, 3),activation=activations.selu, padding='same', kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=activations.selu, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(input, output)
    # model.compile(loss=losses.categorical_crossentropy,
    #               optimizer=optimizers.Adam(lr=0.002),
    #               metrics=['acc'])
    return model


def model2(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)        
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)        
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(_input, output)
    return model


def model3(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)        
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(_input, output)
    return model

def model4(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)        
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(_input, output)
    return model


def model5(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(_input, output)
    return model

def model6(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=(4, 4), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(_input)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=(4, 4), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    
def model7(input_shape, num_classes):    
    _input = layers.Input(shape=input_shape)
    x = layers.Flatten()(_input)
    x = layers.Dense(1024*4, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(_input, output)
    return model

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        # attention takes three inputs: queries, keys, and values,
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        # use the product between the queries and the keys 
        # to know "how much" each element is the sequence is important with the rest
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        # resulting vector, score is divided by a scaling factor based on the size of the embedding
        # scaling fcator is square root of the embeding dimension
        scaled_score = score / tf.math.sqrt(dim_key)
        # the attention scaled_score is then softmaxed
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # Attention(Q, K, V ) = softmax[(QK)/√dim_key]V
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
         
        batch_size = tf.shape(inputs)[0]
        #MSA takes the queries, keys, and values  as input from the previous layer 
        #and projects them using the three linear layers.
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        #self attention of different heads are concatenated  
        output = self.combine_heads(concat_attention)
        return output
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Transfromer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        out1 = self.layernorm1(inputs)       
        attention_output = self.multiheadselfattention(out1)
        attention_output = self.dropout1(attention_output, training=training)       
        out2 = self.layernorm2(inputs + attention_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return (out2 + ffn_output)

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        # create patches based on patch_size
        # image_size/patch_size==0
        self.dim = (image_size[0], image_size[1], channels)
        num_patches = self.create_patch(image_size, patch_size, channels)
        self.reshape_layer = tf.keras.layers.Reshape((None, 128, 192, 1), input_shape=(None, 128, 32, 6))
        self.d_model = d_model
        self.patch_proj = self.create_postional_embedding(num_patches, d_model)
        self.enc_layers = [ TransformerBlock(d_model, num_heads, mlp_dim, dropout) for _ in range(num_layers) ]
        self.mlp_head = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation=tfa.activations.gelu),
                layers.Dropout(dropout),
                layers.Dense(num_classes, name='y'),
            ]
        )
        
    def substract_mean(self, x):
        x_mean = tf.repeat(tf.math.reduce_mean(x, axis=3, keepdims=True), 6, axis=3)
        return x - x_mean
        
    def normalize(self, x):
        return ((x-tf.math.reduce_min(x, axis=[1, 2, 3], keepdims=True)) / (tf.math.reduce_max(x, axis=[1, 2, 3], keepdims=True) - tf.math.reduce_min(x, axis=[1, 2, 3], keepdims=True))*2)-1
    
    def create_patch(self, image_size, patch_size, channels):
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_dim = channels * patch_size[0] * patch_size[1]
        self.patch_size = patch_size
        print("num_patches:", num_patches)
        return num_patches
    
    def create_postional_embedding(self, num_patches, d_model):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        print("positional embedding:", self.pos_emb.shape)
        print("class embedding:", self.class_emb.shape)
        return layers.Dense(d_model)
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        ) 
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        # rescale
        x = self.substract_mean(x)
        x = self.normalize(x)
        
        # x = self.reshape_layer(x)
        # extract the patches from the image
        patches = self.extract_patches(x)
        # apply the postional embedding
        x = self.patch_proj(patches)
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb        
        for layer in self.enc_layers:
            x = layer(x, training)
        x = self.mlp_head(x[:, 0])
        return x
    
    def build_graph(self):
        x = layers.Input(shape=self.dim)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
def modelTransformer(input_shape, num_classes):    
    IMAGE_SIZE=input_shape
    PATCH_SIZE=(16, 16)
    NUM_LAYERS=12
    NUM_HEADS=12
    MLP_DIM=3072
    D_MODEL=768
        
    net = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        channels=6,
        dropout=0.1,
        )
    return net