
"""
モデルをいくつか入れて、切り替えられるようにする
"""
import numpy as np
import pandas as pd
import os, sys, time, csv
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from PIL import Image
import glob



import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np



class model():
	"""
	def __init__(self, INPUT_SIZE, CLASS_NUM, MODEL_MODE):
		self.INPUT_SIZE = INPUT_SIZE
		self.CLASS_NUM = CLASS_NUM
		self.MODEL_MODE = MODEL_MODE
		
		print("ここ")
		print(CLASS_NUM)
		use_model()
	"""
	
	
	def use_model(INPUT_SIZE, COLOR_CHANNEL, CLASS_NUM, MODEL_MODE):
		import numpy as np
		import pandas as pd
		import os, sys, time, csv
		import keras
		from keras.utils import np_utils
		from keras.models import Sequential, load_model
		from keras.layers.convolutional import Conv2D, MaxPooling2D
		from keras.layers.core import Dense, Dropout, Activation, Flatten
		from keras.preprocessing.image import load_img, img_to_array
		from keras.initializers import TruncatedNormal, Constant
		from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
		from keras.optimizers import SGD
		
		from sklearn.model_selection import train_test_split
		from PIL import Image
		import glob		
		
		
		import keras
		from keras.models import Sequential
		from keras.layers.convolutional import Conv2D, MaxPooling2D
		from keras.layers.core import Dense, Dropout, Activation, Flatten
		import numpy as np
		
		from keras.initializers import TruncatedNormal, Constant
		from keras.models import Sequential
		from keras.optimizers import SGD
		from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
		from keras.callbacks import Callback, EarlyStopping
		from keras.utils.np_utils import to_categorical
		
		
		def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
			print("●〇●〇●〇●〇")
			trunc = TruncatedNormal(mean=0.0, stddev=0.01)
			cnst = Constant(value=bias_init)
			return Conv2D(
				filters,
				kernel_size,
				strides=strides,
				padding='same',
				activation='relu',
				kernel_initializer=trunc,
				bias_initializer=cnst,
				**kwargs
			)
		
		def dense(units, **kwargs):		
			trunc = TruncatedNormal(mean=0.0, stddev=0.01)
			cnst = Constant(value=1)
			return Dense(
				units,
				activation='tanh',
				kernel_initializer=trunc,
				bias_initializer=cnst,
				**kwargs
			)		
		
		
		
		model = Sequential()
		print("use_model入った")
		if MODEL_MODE == 0:
			
			# AlexNET_cifar10_INPUT_SIZE_32
			# 第1畳み込み層
			print("1")
			model.add(conv2d(96, 11, strides=(1,1), bias_init=0, input_shape=(INPUT_SIZE, INPUT_SIZE, COLOR_CHANNEL)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			model.add(BatchNormalization())
			
			print("2")
			# 第2畳み込み層
			model.add(conv2d(256, 5, bias_init=1))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			model.add(BatchNormalization())
			
			print("3")
			# 第3～5畳み込み層
			model.add(conv2d(384, 3, bias_init=0))
			model.add(conv2d(384, 3, bias_init=1))
			print("3.2")
			model.add(conv2d(256, 3, bias_init=1))
			print("3.3")
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
			#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
			print("3.4")
			model.add(BatchNormalization())
			
			print("4")
			# 密結合層
			model.add(Flatten())
			model.add(Dense(4096))
			model.add(Dropout(0.5))
			model.add(Dense(4096))
			
			print("5")
			# 読出し層
			model.add(Dense(CLASS_NUM, activation='softmax'))
			
		if MODEL_MODE == 5:
			
			# AlexNET_INPUT_SIZE_224
			# 第1畳み込み層
			print("1")
			#model.add(conv2D(96, 11, strides=(4,4), bias_init=0, input_shape=(INPUT_SIZE, INPUT_SIZE, COLOR_CHANNEL)))
			model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(INPUT_SIZE, INPUT_SIZE, COLOR_CHANNEL)))
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
			model.add(BatchNormalization())
			
			print("2")
			# 第2畳み込み層
			model.add(conv2d(256, 5, bias_init=1))
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
			model.add(BatchNormalization())
			
			print("3")
			# 第3～5畳み込み層
			#model.add(conv2d(384, 3, bias_init=0))
			#model.add(conv2d(384, 3, bias_init=1))
			print("3.2")
			model.add(conv2d(256, 3, bias_init=1))
			print("3.3")
			model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
			#model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering="th"))
			print("3.4")
			model.add(BatchNormalization())
			
			print("4")
			# 密結合層
			model.add(Flatten())
			model.add(Dense(4096))
			model.add(Dropout(0.5))
			model.add(Dense(4096))
			
			print("5")
			# 読出し層
			model.add(Dense(CLASS_NUM, activation='softmax'))
			
		elif MODEL_MODE == 1:
			
			# conv2
			model.add(conv2d(32, (3, 3), padding='same',input_shape=(INPUT_SIZE, INPUT_SIZE, COLOR_CHANNEL)))
			model.add(Activation('relu'))
			model.add(conv2d(32, (3, 3)))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.25))
			
			model.add(conv2d(64, (3, 3), padding='same'))
			model.add(Activation('relu'))
			model.add(conv2d(64, (3, 3)))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.25))
			
			model.add(Flatten())
			model.add(Dense(512))
			model.add(Activation('relu'))
			model.add(Dropout(0.5))
			model.add(Dense(CLASS_NUM))
			model.add(Activation('softmax'))
			
			
		elif MODEL_MODE == 2:
			
			# VGG16
			#input_shape=x_train.shape[1:]
			input_shape=(INPUT_SIZE, INPUT_SIZE, COLOR_CHANNEL)
			
			#model = Sequential()
			model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=input_shape, name='block1_conv1'))
			model.add(BatchNormalization(name='bn1'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='block1_conv2'))
			model.add(BatchNormalization(name='bn2'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block1_pool'))
			model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='block2_conv1'))
			model.add(BatchNormalization(name='bn3'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='block2_conv2'))
			model.add(BatchNormalization(name='bn4'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block2_pool'))
			model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv1'))
			model.add(BatchNormalization(name='bn5'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv2'))
			model.add(BatchNormalization(name='bn6'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='block3_conv3'))
			model.add(BatchNormalization(name='bn7'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block3_pool'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv1'))
			model.add(BatchNormalization(name='bn8'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv2'))
			model.add(BatchNormalization(name='bn9'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block4_conv3'))
			model.add(BatchNormalization(name='bn10'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block4_pool'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv1'))
			model.add(BatchNormalization(name='bn11'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv2'))
			model.add(BatchNormalization(name='bn12'))
			model.add(Activation('relu'))
			model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', name='block5_conv3'))
			model.add(BatchNormalization(name='bn13'))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='block5_pool'))
			model.add(Flatten(name='flatten'))
			model.add(Dense(units=4096, activation='relu', name='fc1'))
			model.add(Dense(units=4096, activation='relu', name='fc2'))
			model.add(Dense(units=CLASS_NUM, activation='softmax', name='predictions'))
		
		
		model.summary()
		return model
		