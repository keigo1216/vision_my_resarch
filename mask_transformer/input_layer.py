import tensorflow as tf
from multi_head_attention import MultiHeadSelfAttention
from encoder import vitEncoderBlock

class cls_token(tf.keras.layers.Layer):
    """
    クラストークンを生成し入力と結合させるレイヤ

    Attributes
    ----------
    emb_dim : int
        埋め込みの次元
    """

    def __init__(self, emb_dim, *args, **kwargs):
        """
        Parameters
        ----------
        emb_dim : int
            埋め込みの次元
        """
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
    
    def build(self, input_shape):
        self.token = self.add_weight(name='cls_token',
                                     shape=[1, 1, self.emb_dim],
                                     trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : Tensor
            形状は(B, N, D)
            B : バッチ、N : トークン数（パッチの個数）、D : 埋め込みベクトルの次元

        Returns
        ----------
        z_0 : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの個数）、D : 埋め込みベクトルの次元

        """
        batch_size = tf.shape(inputs)[0]
        z_0 = tf.concat([tf.tile(self.token, [batch_size, 1 ,1]), inputs], 1)
        return z_0

class pos_emb(tf.keras.layers.Layer):
    """
    位置埋め込みを生成し入力と足し合わせるレイヤ
    
    Attributes
    ----------
    emb_dim : int
        埋め込みの次元
    """
    def __init__(self, emb_dim, num_patch, *args, **kwargs):
        """
        Parameters
        ----------
        emb_dim : int
            埋め込みの次元
        
        num_patch : int
            トークン数（パッチの個数）
        """
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.num_patch = num_patch
    
    def build(self, input_shape):
        self.pos = self.add_weight(name='pos',
                                   shape=[1, self.num_patch + 1, self.emb_dim],
                                   trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの個数)、D : 埋め込みベクトルの次元

        Returns
        ----------
        z_0 : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの個数)、D : 埋め込みベクトルの次元
        """
        z_0 = inputs + self.pos
        return z_0

class VitInputLayer(tf.keras.Model):
    """
    入力画像の埋め込み層。
    クラストークン埋め込み、位置埋め込みも行う。

    Attributes
    ----------
    in_channels : int
            入力画像のチャネル数。
    emb_dim : int
        埋め込みの次元。
    num_patch_row : int
        パッチの列のサイズ
    image_size : int
        画像の列のサイズ
    
    """


    def __init__(self, in_channels:int=1, emb_dim:int=384, num_patch_row:int=2, image_size:int=28, *args, **kwargs):
        """
        Parameters
        ----------
        in_channels : int
            入力画像のチャネル数。
        emb_dim : int
            埋め込みの次元。
        num_patch_row : int
            列方向のパッチの数
        image_size : int
            入力画像の一辺のピクセル数
        """

        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        #パッチの数
        self.num_patch = self.num_patch_row ** 2

        #パッチの一辺のピクセル数
        self.patch_size = self.image_size // self.num_patch_row

        #入力画像のパッチへの分割とパッチの埋め込みを行う層の定義
        self.patch_emb_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.emb_dim, kernel_size=(self.patch_size, self.patch_size), strides=(1, 1), padding="same"),
            tf.keras.layers.MaxPool2D(pool_size=(self.patch_size, self.patch_size))
        ])
        # self.patch_emb_layer = tf.keras.layers.Conv2D(filters=self.emb_dim, kernel_size=(self.patch_size, self.patch_size), strides=(1, 1), padding="same")
        # self.pooling = tf.keras.layers.MaxPool2D(pool_size=(self.patch_size, self.patch_size))

        #cls_tokenを連結するレイヤ
        self.add_cls_token = cls_token(emb_dim=self.emb_dim)

        #positional_embeddingを全体に加算するレイヤ
        self.add_pos_emb = pos_emb(self.emb_dim, self.num_patch)

    def call(self, x):
        """
        Parameters
        ----------
        x : Tensor
            入力画像
            形状は(B, H, W, C)
            B : バッチサイズ、H : 高さ、W : 幅、C : チャネル数
        
        Returns
        ----------
        z_0 : Tensor
            Vitへの入力
            形状は(B, N, D)
            B : バッチサイズ、N : トークン数（パッチの個数）、D : 埋め込みベクトルの次元
        """

        #バッチサイズの取得
        batch_size = tf.shape(x)[0]

        #(batch_size, image_size, image_size, in_channels) -> (batch_size, num_patch_row, num_patch_row, emb_dim)
        z_0 = self.patch_emb_layer(x)

        #(batch_size, num_patch_row, num_patch_row, emb_dim) -> (batch_size, num_patch, emb_dim)
        #精度が上がらなかったらここを疑うポイント
        z_0 = tf.reshape(z_0, [batch_size, self.num_patch, self.emb_dim])

        #cls_tokenを先頭につける
        #(batch_size, num_patch, emb_dim) -> (batch_size, num_patch + 1, emb_dim)
        z_0 = self.add_cls_token(z_0)

        #postional embeddingを全体に行う
        #(batch_size, num_patch, emb_dim) -> (batch_size, num_patch + 1, emb_dim)
        z_0 = self.add_pos_emb(z_0)

        return z_0



if __name__ == '__main__':
    #パラメータの定義
    in_channels = 1
    emb_dim = 384
    num_patch_row = 7
    image_size = 28
    head = 8
    dropout=0.5

    #データセットの生成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = tf.cast(x_train / 255.0, dtype=tf.float32), tf.cast(x_test / 255.0, dtype=tf.float32)
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    model = VitInputLayer(in_channels=in_channels, emb_dim=emb_dim, num_patch_row=num_patch_row, image_size=image_size)
    vit_block = vitEncoderBlock(emb_dim, head, emb_dim*4, dropout)
    out = model(x_train[:100])
    out = vit_block(out)
    print(out.shape)
    