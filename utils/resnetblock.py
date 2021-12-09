from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from numpy.random import choice
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


class ResNetIdentity(Model):
	def __init__(self,kernel_size,filters):
		super(ResNetIdentity,self).__init__()
		self.conv_1 = Conv2D(filters,kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal'
                   ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_1 = BatchNormalization()
		self.ac_1 = Activation(relu)

		self.conv_2 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal'
              ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_2 = BatchNormalization()
		self.ac_2 = Activation(relu)
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
		x = self.add([x,x_prev])
		return self.ac_3(x)


class ResNetDownsampling(Model):
	def __init__(self,kernel_size,filters):
		super(ResNetDownsampling,self).__init__()
		self.conv_1 = Conv2D(filters,kernel_size=kernel_size, strides=(2, 2), padding='same',kernel_initializer='he_normal'
                            ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_1 = BatchNormalization()
		self.ac_1 = Activation(relu)

		self.conv_2 = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same',kernel_initializer='he_normal'
                            ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_2 = BatchNormalization()
		self.ac_2 = Activation(relu)

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
		x_prev = self.conv_3(x_prev)
		x_prev = self.bn_3(x_prev,training=training)
		x = self.add([x,x_prev])
		return self.ac_3(x)

	# def compute_output_shape(self, input_shape):
 #        return (input_shape[0], input_shape[1], input_shape[2])


