import tensorflow as tf

class MultiHeadSelfAttention(tf.keras.Model):
    """
    MultiHeadSelfAttentionの層

    Attributes
    ----------
    """
    def __init__(self, num_patch_row:int, emb_dim:int, head:int, dropout:float, *args, **kwargs):
        """
        Parameters
        ----------
        num_patch : int
            パッチの数
            クラストークンの数を含めていない値
        emb_dim : int
            埋め込みベクトルの次元
        head : int
            分割するheadの大きさ
        dropout : float
            Dropout層のドロップアウト率
        """
        super().__init__(*args, **kwargs)

        self.emb_dim = emb_dim
        self.head = head
        self.head_dim = emb_dim // self.head #一つのヘッドの次元
        self.sqrt_dh = self.head_dim ** 0.5
        
        self.num_patch = num_patch_row ** 2 + 1
        
        #maskを作成するときの線形変換用の層
        self.w_mask = tf.keras.layers.Dense(self.num_patch, use_bias=False)
        #self.create_mask = createMask()

        self.w_q = tf.keras.layers.Dense(emb_dim, use_bias=False)
        self.w_k = tf.keras.layers.Dense(emb_dim, use_bias=False)
        self.w_v = tf.keras.layers.Dense(emb_dim, use_bias=False)

        self.drop_layer = tf.keras.layers.Dropout(dropout)

        self.w_o = tf.keras.Sequential([
            tf.keras.layers.Dense(self.emb_dim),
            tf.keras.layers.Dropout(dropout)
        ])
    
    def call(self, z :tf.Tensor) -> tf.Tensor : 
        """
        Parameters
        ----------
        z : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの数）、D : 埋め込みベクトルの次元

        Returns
        ----------
        out : Tensor
            形状は(B, N+1, D)
            B : バッチ、N : トークン数（パッチの数）、D : 埋め込みベクトルの次元
        """

        #num_patchはclass_tokenまで入れた値で定義する
        #めんどくさいから修正しない
        batch_size, num_patch, _ = tf.unstack(tf.shape(z))
        #batch_size = tf.shape(z)[0]
        #num_patch = tf.shape(z)[1]

        #(batch_size, num_patch+1, emb_dim)->(batch_size, num_patch+1, emb_dim)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)
        
        
        #maskを作成する
        #クエリからマスクを作る（他にいい方法があるかもしれないが、一番簡単に思いつく方法で）
        #(batch_size, num_patch+1, emb_dim) -> (batch_size, num_patch+1, num_patch+1)
        mask = self.w_mask(q)
        mask = tf.nn.softmax(mask, axis=-1)
        
        #maskの値を1->0, 0->マイナス無限大にする
        epsilon = 1e-6 #めっちゃ小さい値にしておけばOK
        #0.5は閾値的なイメージ
        #ここが大きければAttentionの範囲が小さくなる
        #ここが大きければ（プラスならば）Attentionの範囲は大きくなる
        #thresholdはハイパーパラメータ、こんなところに書くな！
        threshold = 0.2
        mask = (mask - threshold) / epsilon
        #マイナスだけ垂れ流してプラスは0にする
        #tf.whereを使いたかったけど勾配が流れないっぽいから力技
        mask = -tf.nn.relu(-mask)

        
        #軸を追加してMultiHeadに対応する
        #(batch_size, num_patch+1, num_patch+1) -> (batch_size, head, num_patch+1, num_patch+1)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, self.head, 1, 1])


        #headに分ける
        #(batch_size, num_patch+1, emb_dim)->(batch_size, num_patch+1, head, emb_dim // head)
        q = tf.reshape(q, [batch_size, num_patch, self.head, self.emb_dim // self.head])
        k = tf.reshape(k, [batch_size, num_patch, self.head, self.emb_dim // self.head])
        v = tf.reshape(v, [batch_size, num_patch, self.head, self.emb_dim // self.head])

        #(batch_size, num_patch+1, head, emb_dim // head)->(batch_size, head, num_patch+1, emb_dim//head)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        dots = tf.matmul(q, k, transpose_b=True) / self.sqrt_dh  #(batch_size, head, num_patch+1, emb_dim//head)*(batch_size, head, emb_dim//head, num_patch+1)->(batch_size, head, num_patch+1, num_patch+1)
        attn = tf.nn.softmax(dots+mask, axis=-1) #(batch_size, head, num_patch+1, num_patch+1)->(batch_size, head, num_patch+1, num_patch+1)
        attn = self.drop_layer(attn) #(batch_size, head, num_patch+1, num_patch+1)->(batch_size, head, num_patch+1, num_patch+1)

        out = tf.matmul(attn, v) #(batch_size, head, num_patch+1, num_patch+1)*(batch_size, head, num_patch+1, emb_dim//head)->(batch_size, head, num_patch+1, emb_dim//head)
        
        out = tf.transpose(out, [0, 2, 1, 3]) #(batch_size, num_patch+1, head, emb_dim//head)
        out = tf.reshape(out, [batch_size, num_patch, self.emb_dim]) #(batch_size, num_patch+1, head, emb_dim//head)->(batch_size, num_patch+1, emb_dim)

        out = self.w_o(out) #(batch_size, num_patch+1, emb_dim)->(batch_size, num_patch+1, emb_dim)
        return out
    
# class createMask(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
        
#     def build(self, input_shape):
        
#         self.threshold = self.add_weight(
#             shape = [1],
#             dtype=tf.float32,
#             initializer=tf.keras.initializers.Zeros()
#         )
        
#         super().build(input_shape)
    
#     def call(self, mask):
#         return tf.where(mask > self.threshold, 1, 0)