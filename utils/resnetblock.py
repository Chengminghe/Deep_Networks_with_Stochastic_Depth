from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from numpy.random import choice
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


class TwoLayerBlock(Model):
	def __init__(self,kernel_size,filters,down_sampling = False):
		super(TwoLayerBlock,self).__init__()
		self.down_sampling = down_sampling
		stride = (1,1)
		if self.down_sampling:
			stride = (2,2)
		self.conv_1 = Conv2D(filters,kernel_size=kernel_size, strides=stride, padding='same',kernel_initializer='he_normal'
                   ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_1 = BatchNormalization()
		self.ac_1 = Activation(relu)

		self.conv_2 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal'
              ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_2 = BatchNormalization()
		self.ac_2 = Activation(relu)
		if down_sampling:
			self.conv_3 = Conv2D(filters, kernel_size=(1,1), strides=(2, 2), padding='same',kernel_initializer='he_normal'
	                ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
			self.bn_3 = BatchNormalization()
		self.add = Add()
		self.ac_3 = Activation(relu)
		
	def call(self,x,training=False):
		x_prev = x
		x = self.conv_1(x)
		x = self.bn_1(x,training=training)
		x = self.ac_1(x)
		x = self.conv_2(x)
		x = self.bn_2(x,training=training)
		x = self.ac_2(x)
		if self.down_sampling:
			x_prev = self.conv_3(x_prev)
			x_prev = self.bn_3(x_prev,training=training)
		x = self.add([x,x_prev])
		return self.ac_3(x)

class ThreeLayerBlock(Model):
	def __init__(self,filters,down_sampling = False):
		super(ThreeLayerBlock,self).__init__()
		self.f1, self.f2, self.f3 = filters
		self.down_sampling = down_sampling
		stride = (1,1)
		if self.down_sampling:
			stride = (2,2)
		self.conv_1 = Conv2D(self.f1,kernel_size=(1,1), strides=stride,padding='same',kernel_initializer='he_normal'
                            ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_1 = BatchNormalization()
		self.ac_1 = Activation(relu)
		self.conv_2 = Conv2D(self.f2,kernel_size=(3,3), padding='same',kernel_initializer='he_normal'
                    		,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_2 = BatchNormalization()
		self.ac_2 = Activation(relu)
		self.conv_3 = Conv2D(self.f3,kernel_size=(1,1), padding='same',kernel_initializer='he_normal'
                    		,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_3 = BatchNormalization()
		self.ac_3 = Activation(relu)
		self.conv_4 = Conv2D(self.f3,kernel_size=(1,1), strides=stride, padding='same',kernel_initializer='he_normal'
	                ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_4 = BatchNormalization()
		self.add = Add()
		self.ac_4 = Activation(relu)


	def call(self,x,training=False):
		x_prev = x
		x = self.conv_1(x)
		x = self.bn_1(x,training=training)
		x = self.ac_1(x)
		x = self.conv_2(x)
		x = self.bn_2(x,training=training)
		x = self.ac_2(x)
		x = self.conv_3(x)
		x = self.bn_3(x,training=training)
		x = self.ac_3(x)
		x_prev = self.conv_4(x_prev)
		x_prev = self.bn_4(x_prev,training=training)
		x = self.add([x,x_prev])
		return self.ac_4(x)