import tensorflow as tf
from input_layer import VitInputLayer
from multi_head_attention import MultiHeadSelfAttention

if __name__ == "__main__":
    in_channels = 3
    num_classes = 100
    emb_dim = 384
    num_patch_row = 8
    image_size = 32
    num_block = 6
    head = 8
    hidden_dim = emb_dim * 4
    dropout=0.4
    batch_size = 50
    epochs = 100

    #データセットの生成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train, x_test = tf.cast(x_train / 255.0, dtype=tf.float32), tf.cast(x_test / 255.0, dtype=tf.float32)
