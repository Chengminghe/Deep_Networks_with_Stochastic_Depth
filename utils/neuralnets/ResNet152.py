from utils.resnetblock import TwoLayerBlock
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D,BatchNormalization,Activation, Add,Flatten,ZeroPadding2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
class ResNet152(Model):
	def __init__(self,input_shape,num_class):
		super(ResNet152,self).__init__()
		self.conv_0 = Conv2D(16, kernel_size=(3, 3),kernel_initializer='he_normal',input_shape= input_shape
			,padding='same',bias_initializer='zeros',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_0 = BatchNormalization()
		self.ac_0 = Activation(relu)
		self.resblocks = dict()
		self.resblocks['res_1_1'] = TwoLayerBlock((3,3),16)
		self.resblocks['res_2_1'] = TwoLayerBlock((3,3),32,down_sampling = True)
		self.resblocks['res_3_1'] = TwoLayerBlock((3,3),64,down_sampling = True)
		for i in range(24):
			self.resblocks['res_1_%d'%(i+2)] = TwoLayerBlock((3,3),16)
			self.resblocks['res_2_%d'%(i+2)] = TwoLayerBlock((3,3),32)
			self.resblocks['res_3_%d'%(i+2)] = TwoLayerBlock((3,3),64)
		self.avg = GlobalAveragePooling2D()
		self.fc = Dense(num_class,activation='softmax', kernel_initializer='he_normal'
             ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))



	def call(self,x,training=False):
		x = self.conv_0(x)
		x = self.bn_0(x,training=training)
		x = self.ac_0(x)
		for i in range(3):
			for j in range(25):
				x = self.resblocks['res_%d_%d'%((i+1),(j+1))](x,training=training)
		x = self.avg(x)
		return self.fc(x)