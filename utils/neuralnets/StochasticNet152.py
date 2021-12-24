from utils.stochasticblock import TwoLayerStochastic
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D,BatchNormalization,Activation, Add,Flatten,ZeroPadding2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
class StochasticNet152(Model):
	def __init__(self,input_shape,num_class,p_L = 0.5):
		super(StochasticNet152,self).__init__()
		self.p_L = p_L
		self.L = 75
		self.conv_0 = Conv2D(16, kernel_size=(3, 3),kernel_initializer='he_normal',input_shape= input_shape
			,padding='same',bias_initializer='zeros',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))
		self.bn_0 = BatchNormalization()
		self.ac_0 = Activation(relu)
		self.resblocks = dict()
		self.resblocks['res_1_1'] = TwoLayerStochastic((3,3),16,keep_prob=(1-1/self.L*(1-self.p_L)))
		self.resblocks['res_2_1'] = TwoLayerStochastic((3,3),32,keep_prob=(1-26/self.L*(1-self.p_L)),down_sampling=True)
		self.resblocks['res_3_1'] = TwoLayerStochastic((3,3),64,keep_prob=(1-51/self.L*(1-self.p_L)),down_sampling=True)
		for i in range(24):
			self.resblocks['res_1_%d'%(i+2)] = TwoLayerStochastic((3,3),16,keep_prob=(1-(i+2)/self.L*(1-self.p_L)))
			self.resblocks['res_2_%d'%(i+2)] = TwoLayerStochastic((3,3),32,keep_prob=(1-(i+27)/self.L*(1-self.p_L)))
			self.resblocks['res_3_%d'%(i+2)] = TwoLayerStochastic((3,3),64,keep_prob=(1-(i+52)/self.L*(1-self.p_L)))
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