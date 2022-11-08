import tensorflow as tf
from multi_head_attention import MultiHeadSelfAttention

class vitEncoderBlock(tf.keras.Model):
    """
    エンコーダのブロック

    Attributes
    ----------
    ln1 : tensor
        正則化の層
    msa : tensor
        multiheadattentionの層
    ln2 : tensor
        正則化の層
    mlp : tensor
        多層パーセプトロンの層
    """
    def __init__(self, num_patch_row:int, emb_dim:int, head:int, hidden_dim:int, dropout:float, *args, **kwargs):
        """
        Parameters
        ----------
        emb_dim : int
            埋め込みベクトルの次元
        head : int
            分割するheadの大きさ
        hidden_dim : int
            MLPにおける中間層の次元
        dropout : float
            Dropout層のドロップアウト率
        """
        super().__init__(*args, **kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.msa = MultiHeadSelfAttention(num_patch_row=num_patch_row, emb_dim=emb_dim, head=head, dropout=dropout)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout), #テストの時はデフォルトでOFFになる
            tf.keras.layers.Dense(emb_dim),
            tf.keras.layers.Dropout(dropout)
        ])
    
    def call(self, z:tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        z : tf.Tensor
            形状(B, N+1, D)
            B : バッチ、N : トークン数（パッチの数）、D : 埋め込みベクトルの次元

        Returns
        ----------
        out : tf.Tensor
            形状(B, N+1, D)
            B : バッチ、N : トークン数（パッチの数）、D : 埋め込みベクトルの次元
        """
        out = self.msa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out
        return out