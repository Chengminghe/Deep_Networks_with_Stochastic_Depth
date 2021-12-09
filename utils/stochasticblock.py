from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from numpy.random import choice
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from utils.resnetblock import ResNetIdentity, ResNetDownsampling


class StochasticIdentity(ResNetIdentity):
    def __init__(self,kernel_size,filters,keep_prob):
        super(StochasticIdentity,self).__init__(kernel_size,filters)
        self.keep_prob = keep_prob

        
    def call(self,x,training=False):
        if training == True:
            if choice(2,p=[1-self.keep_prob,self.keep_prob]) == 1:
                return super().call(x,training=training)
            else:
                return Activation(relu)(x)
        else:
            return super().call(x,training=training)


class StochasticDownsampling(ResNetDownsampling):
    def __init__(self,kernel_size,filters,keep_prob):
        super(StochasticDownsampling,self).__init__(kernel_size,filters)
        self.keep_prob = keep_prob

    def call(self,x,training=False):
        if training == True:
            if choice(2,p=[1-self.keep_prob,self.keep_prob]) == 1:
                return super().call(x,training=training)
            else:
                x = self.conv_3(x)
                x = self.bn_3(x,training=training)
                return Activation(relu)(x)
        else:
            return super().call(x,training=training)
    


