【要点まとめ】  
画像を用いた訓練を実施する際に、大量の訓練データが必要となるが  
それらを一つ一つ用意するのは現実的に難しい。  
そのため、data_augmentationと呼ばれる水増し手法が一般的に使われる。
※ただし画像によっては適さない手法があるので注意が必要。（6と9の反転や8と∞の回転）
  
【実装演習】  
tensorflowには水増しを実施するための関数が多数用意されているので、それらを実装する  
```
# Horizontal Flip
image = tf.image.random_flip_left_right(image, seed=123)
# Vertical Flip
image = tf.image.random_flip_up_down(image, seed=123)
# Crop
image = tf.image.random_crop(image, size=(100, 100, 3), seed=123)
# Contrast
# コントラスト強弱の下限と上限を指定可能
image = tf.image.random_contrast(image, lower=0.4, upper=0.6)
# Brightness
# 輝度の範囲を指定可能
image = image = tf.image.random_brightness(image, max_delta=0.8)
# Hue
# 色相の範囲を指定可能
image = tf.image.random_hue(image, max_delta=0.1)
# Rotate
# 以下のコードは90度回転
image = tf.image.rot90(image, k=1)
```
  
【実装演習考察】  
画像の水増し手法は知っていましたが、tensorflowのコード1行で実装できるのは知りませんでした。  
サンプルコードにもありましたが、複数の手法を組み合わせる関数を用意することで、ほぼ無限に画像を用意できるのは大変便利だと感じました。  
Rotateなど複数の関数が用意されている場合もあるので、対象によって使い分けていきます。  
