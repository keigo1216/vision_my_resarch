import tensorflow as tf
from input_layer import VitInputLayer
from encoder import vitEncoderBlock
import matplotlib.pyplot as plt
import pandas as pd
#from dataset import load_dataset
#from sklearn.model_selection import train_test_split

class vit(tf.keras.Model):
    """
    vision transformerのクラス

    Attributes
    ----------
    input_layer : tensor
        入力画像を埋め込みする層
    encoder : tensor
        encoderの層
    mlp : tensor
        多層パーセプトロンの層

    """
    def __init__(self,
        in_channles:int,
        num_classes:int,
        emb_dim:int,
        num_patch_row:int,
        image_size:int,
        num_blocks:int,
        head:int,
        hidden_dim:int,
        dropout:int,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        
        """
        Parameters
        in_channels : int
            入力画像のチャネル数。
        num_classes : int
            分類するクラスの数
        emb_dim : int
            埋め込みの次元。
        num_patch_row : int
            列方向のパッチの数
        image_size : int
            入力画像の一辺のピクセル数
        num_blocks : int
            Encoderブロックの数
        head : int
            分割するheadの大きさ
        hidden_dim : int
            MLPの隠れ層の次元
        dropout : float
            Dropout層のドロップアウト率
        """

        self.num_classes = num_classes
        
        self.input_layer = VitInputLayer(in_channels=in_channles,
                            emb_dim=emb_dim,
                            num_patch_row=num_patch_row,
                            image_size=image_size)

        self.encoder = tf.keras.Sequential([
            vitEncoderBlock(emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
                )
                for _ in range(num_blocks)])

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])


    def call(self, x:tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        z : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの数）、D : 埋め込みベクトルの次元
        
        Returns
        ----------
        pred : Tensor
            形状は(B, M)
            B : バッチ、M : クラス数
        """

        out = self.input_layer(x) #(batch_size, H, W, C)->(batch_size, N+1, D)
        out = self.encoder(out) #(batch_size, N+1, D)->(batch_size, N+1, D)
        pred = self.mlp(out[:, 0, :]) #(batch_size, 1, D)->(batch_size, M)
        return pred

def mnist():
    #パラメータの定義
    in_channels = 1
    num_classes = 10
    emb_dim = 384
    num_patch_row = 7
    image_size = 28
    num_block = 6
    head = 8
    hidden_dim = emb_dim * 4
    dropout=0.5
    batch_size = 100
    epochs = 100

    #データセットの生成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = tf.cast(x_train / 255.0, dtype=tf.float32), tf.cast(x_test / 255.0, dtype=tf.float32)
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    # vit_model = vit(in_channles=in_channels,
    #                 num_classes=num_classes,
    #                 emb_dim=emb_dim,
    #                 num_patch_row=num_patch_row,
    #                 image_size=image_size,
    #                 num_blocks=num_block,
    #                 head=head,
    #                 hidden_dim=hidden_dim,
    #                 dropout=dropout)
    # vit_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(1e-04),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']

    # )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = vit_model = vit(in_channles=in_channels,
                    num_classes=num_classes,
                    emb_dim=emb_dim,
                    num_patch_row=num_patch_row,
                    image_size=image_size,
                    num_blocks=num_block,
                    head=head,
                    hidden_dim=hidden_dim,
                    dropout=dropout)
        
        vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-04),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    vit_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)

def cifar100():
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


    # vit_model = vit(in_channles=in_channels,
    #                 num_classes=num_classes,
    #                 emb_dim=emb_dim,
    #                 num_patch_row=num_patch_row,
    #                 image_size=image_size,
    #                 num_blocks=num_block,
    #                 head=head,
    #                 hidden_dim=hidden_dim,
    #                 dropout=dropout)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = vit_model = vit(in_channles=in_channels,
                    num_classes=num_classes,
                    emb_dim=emb_dim,
                    num_patch_row=num_patch_row,
                    image_size=image_size,
                    num_blocks=num_block,
                    head=head,
                    hidden_dim=hidden_dim,
                    dropout=dropout)
        
        vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-04),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    # vit_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(1e-04),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    history = vit_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)
    vit_model.save_weights('vit_model_drop_dims')

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('vit_model_drop_dims.csv')

    # 可視化
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("vit_model_drop_dims.png")

if __name__ == '__main__':
    cifar100()