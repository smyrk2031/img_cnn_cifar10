
"""
めも書き
①cifer10を使って基本的なCNNの実装をする
②いろいろ実験してみる(パラメータ、アーキテクチャ)
③fine-tuning
④
⑤
⑥

#デバッガ
import pdb; pdb.set_trace()

"""

#各種ライブラリのインポート
import numpy as np
import pandas as pd
import os, sys, time, csv, datetime
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
import glob

import tensorflow as tf
import keras as K
from keras.utils import np_utils, to_categorical
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import models, layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import CNN_models


# ファインチューニング用に使えるImageNetで学習した重みをもつ画像分類のモデル
#keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.mobilenet import MobileNet
#from keras.applications.densenet import DenseNet121	#121か169か201
#from keras.applications.nasnet import NASNetLarge	#NASNetLargeかNASNetMobile
#from keras.applications.mobilenet_v2 import MobileNetV2


# -------------------------------------------------------------------------------------
#                        初期設定部
# -------------------------------------------------------------------------------------
T1 = time.time()
# 実行ファイルの絶対パスを取得(モデル保存用に)
dir = os.path.dirname(os.path.abspath(__file__))
print(dir)
# ファイルの実行時刻を取得(モデルに名前つける為に)
dt_now = datetime.datetime.now()
NewDir = dt_now.strftime("%m%d%H%M")
print(NewDir)


#●〇●〇●〇●〇
# GPUの設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

# GPUメモリをクリアする(学習済modelをクリアする)
K.backend.clear_session()


#●〇●〇●〇●〇
# 学習用の設定
EPOCHS = 8				#学習のEpoch数
BATCH_SIZE = 512			#学習中のBatchSize
VALID_SPLIT = 0.1			#学習中のtrain-data分割割合
opt = "adam"				#最適化関数は何使う？"SDG","adam","RMSprop"
LOSS = "categorical_crossentropy"	#損失関数はどうする？"categorical_crossentropy","",""
METRICS = ["accuracy"]			#最適化する計量は？"accuracy","",""


#●〇●〇●〇●〇
#CIFAR-10のデータセットのインポート
from keras.datasets import cifar10		#データセット詳細：<https://www.cs.toronto.edu/~kriz/cifar.html>
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#================================================
#CIFAR-10の正規化工程
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)

# 特徴量の正規化
X_train = X_train/255.
X_test = X_test/255.

# クラスラベルの変換フォーマットを作成

# ラベル名をリスト化
label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

"""
print(label_names)
print(label_names[0])
print("もと形状")
for i in range(8):
	#print(Y_test[i][0])
	label = Y_test[i][0]
	print(label, "：", label_names[label])
"""

# クラスラベルの1-hotベクトル化
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
"""
#print(Y_test[0])
#print(Y_test.shape)
print("Y_testのラベル確認")
for i in range(8):
	print(Y_test[i])
	label = np.argmax(Y_test[i])
	print(label, "：", label_names[label])
print("表示完了")
"""
#================================================

# テクニック：X_train.shapeは(50000, 32, 32, 3)だから、X_train.shape[1:]で画像サイズが取得できる
#INPUT_SIZE = X_train.shape[1:]
print(X_train.shape[1:])
print(X_train.shape[:1])
print(X_train.shape[1])
print(X_train.shape[3])

print(X_train[0].shape)

#●〇●〇●〇●〇
# モデルを読み込む前の初期設定
FINE_TUNING = 0			#0ならしない、1ならする
INPUT_SIZE = X_train.shape[1]
COLOR_CHANNEL = X_train.shape[3]
CLASS_NUM = 10
MODEL_MODE = 2
CLASS_NUM = len(label_names)

print(INPUT_SIZE)
print(COLOR_CHANNEL)
print(CLASS_NUM)
print(MODEL_MODE)
print(CLASS_NUM)

# モデルを読み込む
if FINE_TUNING == 1:
	input = Input(shape=X_train[0].shape)
	model_base = VGG16(include_top=False, weights="imagenet", input_tensor=input, classes=CLASS_NUM)
	#model = VGG16(include_top=False, weights="imagenet", input_shape=input)
	model_name = "VGG16-fine"
	print("model取得完了")
	model_base.summary()
	model_base.save(dir + "\\" + model_name + "_ImageNet" + ".h5")	
	
	"""
	# 畳み込みベースのvgg16を凍結
	print("凍結")
	model_base.trainable = False
	model_base.summary()
	
	# 全結合層の追加
	print("全結合層の追加")
	model = models.Sequential()
	model.add(model_base)
	model.add(layers.Flatten(input_shape=(1,1,512)))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(CLASS_NUM, activation='softmax'))
	
	
	"""
	# 畳み込みベースのvgg16を凍結せずに再学習層を残す
	print("凍結するけど、再学習層を残す！")
	model_base.trainable = True
	
	set_flg = False
	for layer in model_base.layers:
		if layer.name == "block5_conv1":  #"block5_conv1"というレイヤー以下はすべて trainable を True とする。
			set_flg = True
	
		if set_flg == True:
			layer.trainable = True
			print("set trainable True 凍結しない！")
		else:
			layer.trainable = False
			print("set trainable False 凍結！")
	# 全結合層の追加
	print("全結合層の追加")
	model_base.summary()
	model = models.Sequential()
	model.add(model_base)
	model.add(layers.Flatten(input_shape=(1,1,512)))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(CLASS_NUM, activation='softmax'))
	#"""
	
	print("FineTuneing!!")
	
elif FINE_TUNING == 0:
	model = CNN_models.model.use_model(INPUT_SIZE, COLOR_CHANNEL, CLASS_NUM, MODEL_MODE)
	if MODEL_MODE == 2:
		model_name = "VGG16"
		print("VGG16")
	elif MODEL_MODE == 0:
		print("AlexNet")
		model_name = "AlexNet"

model.summary()
#end


#============================================
# ここから学習工程
#============================================
# コンパイル
model.compile(loss = LOSS, optimizer = opt, metrics = METRICS)

#訓練
print("ここから訓練")
es = EarlyStopping(monitor="accuracy", min_delta=0.000, patience=20, verbose=0, mode='auto')
#history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, validation_split=VALID_SPLIT, callbacks=[es], shuffle=True)
history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, validation_split=VALID_SPLIT, shuffle=True)
print("訓練の所要時間は" + str(time.time()-T1))

# 実行履歴 (正解率の推移) を CSV で保存
print("historyの保存")
loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

with open(dir + "/history_" + model_name + "_epoch" + str(EPOCHS) + ".csv", "wt", encoding="utf-8") as out:
	writer = csv.writer(out)
	writer.writerow(["EPOCH", "ACC(TRAIN)", "ACC(TEST)", "LOSS(TRAIN)", "LOSS(TEST)"])
	for i in range(len(loss)):
		writer.writerow([i+1, acc[i], val_acc[i], loss[i], val_loss[i]])

# モデルの保存
#model.save(dir + model_name + "_epoch" + str(EPOCHS) + ".h5")
#model.save(dir + "\\" + NewDir + "\\" + model_name + "_epoch" + str(EPOCHS) + ".h5")
model.save(dir + "\\" + model_name + "_epoch" + str(EPOCHS) + ".h5")



# =============================================
# 評価 & 評価結果出力
# =============================================
print("評価結果")
score = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
print(score)


# 結果の可視化
print("結果の可視化")
plt.figure(figsize=(10,7))
plt.plot(history.history['accuracy'], color='b', linewidth=3)
plt.plot(history.history['val_accuracy'], color='r', linewidth=3)
plt.tick_params(labelsize=18)
plt.ylabel('acuuracy', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['training', 'test'], loc='best', fontsize=20)
plt.figure(figsize=(10,7))
plt.plot(history.history['loss'], color='b', linewidth=3)
plt.plot(history.history['val_loss'], color='r', linewidth=3)
plt.tick_params(labelsize=18)
plt.ylabel('loss', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['training', 'test'], loc='best', fontsize=20)
plt.show()



# 混合行列の表示
print("混合行列")
Y_pred = model.predict(X_test, batch_size=32, verbose=1)
print("Y_pred形状(テストデータの予測結果)：",Y_pred.shape)
Y_pred_label = np.argmax(Y_pred[0])
print(Y_pred_label)
print(Y_pred_label, "：", label_names[0])
print("ちなみに正解は：",Y_test[0])

yyy = np.argmax(Y_pred, axis=1)
print(Y_pred[0])
print(yyy.shape)
print(yyy[0])

Y_test = np.argmax(Y_test, axis=1)

for i in range(20):
	print("予測：", yyy[i] ,", 正解：",Y_test[i])

print(confusion_matrix(Y_test, yyy))
#tp, fn, fp, tn = confusion_matrix(Y_test, yyy).ravel()
#print("tp, fn, fp, tn")
#print(tp, fn, fp, tn)

print(classification_report(Y_test, yyy))

accuracy_score = accuracy_score(Y_test, yyy, normalize=False)
precision_score = precision_score(Y_test, yyy, average='macro')
recall_score = recall_score(Y_test, yyy, average='macro')
f1_score = f1_score(Y_test, yyy, average='macro')

print("accuracy_score:",accuracy_score)
print("precision_score:",precision_score)
print("recall_score:",recall_score)
print("f1_score:",f1_score)

# 混合行列をヒートマップで出力
print("混合行列をヒートマップで表示")
def print_cmx(Y_test, yyy):
    labels = sorted(list(set(Y_test)))
    cmx_data = confusion_matrix(Y_test, yyy, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True)
    plt.show()

print_cmx(Y_test, yyy)
# GPUメモリをクリアする(学習済modelをクリアする)
print("GPUメモリをクリアします")
K.backend.clear_session()
print("所要時間は" + str(time.time()-T1))
print("終了！")
