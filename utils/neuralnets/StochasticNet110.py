from utils.stochasticblock import StochasticIdentity, StochasticDownsampling
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D,BatchNormalization,Activation, Add,Flatten,ZeroPadding2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
class StochasticNet110(Model):
	def __init__(self,input_shape,num_class,p_L = 0.5):
		super(StochasticNet110,self).__init__()
		self.p_L = p_L
		self.L = 54
		self.conv_0 = Conv2D(16, kernel_size=(3, 3),kernel_initializer='he_normal',input_shape= input_shape
			,padding='same',bias_initializer='zeros',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_0 = BatchNormalization()
		self.ac_0 = Activation(relu)
		self.resblocks = dict()
		self.resblocks['res_1_1'] = StochasticIdentity((3,3),16,keep_prob=(1-1/54*(1-self.p_L)))
		self.resblocks['res_2_1'] = StochasticDownsampling((3,3),32,keep_prob=(1-19/54*(1-self.p_L)))
		self.resblocks['res_3_1'] = StochasticDownsampling((3,3),64,keep_prob=(1-37/54*(1-self.p_L)))
		for i in range(17):
			self.resblocks['res_1_%d'%(i+2)] = StochasticIdentity((3,3),16,keep_prob=(1-(i+2)/54*(1-self.p_L)))
			self.resblocks['res_2_%d'%(i+2)] = StochasticIdentity((3,3),32,keep_prob=(1-(i+20)/54*(1-self.p_L)))
			self.resblocks['res_3_%d'%(i+2)] = StochasticIdentity((3,3),64,keep_prob=(1-(i+38)/54*(1-self.p_L)))
		self.avg = GlobalAveragePooling2D()
		self.fc = Dense(num_class,activation='softmax', kernel_initializer='he_normal'
             ,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))

	def call(self,x,training=False):
		x = self.conv_0(x)
		x = self.bn_0(x,training=training)
		x = self.ac_0(x)
		for i in range(3):
			for j in range(18):
				x = self.resblocks['res_%d_%d'%((i+1),(j+1))](x,training=training)
		x = self.avg(x)
		return self.fc(x)