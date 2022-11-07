import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class convMixerBlock(tf.keras.Model):
    def __init__(self, emb_dim):
        super().__init__()

        self.depth_wise_conv = tf.keras.layers.Conv2D(
            filters=emb_dim,
            kernel_size=(9, 9),
            groups=emb_dim,
            padding="same"
        )
        self.norm1 = tf.keras.layers.BatchNormalization()

        self.point_wise_conv = tf.keras.layers.Conv2D(
            filters=emb_dim,
            kernel_size=(1, 1)
        )
        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        res = input

        x = self.depth_wise_conv(input)
        x = tf.keras.activations.gelu(x)
        x = self.norm1(x)
        x = x + res

        x = self.point_wise_conv(x)
        x = tf.keras.activations.gelu(x)
        out = self.norm2(x)

        return out

class convMixer(tf.keras.Model):
    def __init__(self, patch_size, emb_dim, num_blocks, num_class):
        super().__init__()


        self.emb_layer = tf.keras.layers.Conv2D(
            filters=emb_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size)
        )

        self.conv_mixer_layer = tf.keras.Sequential([
            convMixerBlock(emb_dim) for _ in range(num_blocks)
        ])

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.mlp = tf.keras.layers.Dense(num_class)

    def call(self, input):
        x = self.emb_layer(input)
        x = self.conv_mixer_layer(x)
        x = self.global_pooling(x)
        x = self.flatten(x)
        out = self.mlp(x)
        return out

if __name__ == "__main__":
    num_class = 10
    emb_dim = 384
    patch_size = 1
    num_blocks = 6
    batch_size = 50
    epochs = 1

    #データセットの生成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #x_train, x_test = tf.cast(x_train / 255.0, dtype=tf.float32), tf.cast(x_test / 255.0, dtype=tf.float32)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = convMixer(
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_blocks=num_blocks,
            num_class=num_class
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-04),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)
    model.save_weights('convMixer')

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('convMixer.csv')

    # 可視化
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.plot(epochs, acc, 'bo', label='training acc')
    plt.plot(epochs, val_acc, 'b', label='validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    plt.savefig("convMixer_acc.png")

    plt.figure()

    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.savefig("convMixer_loss.png")