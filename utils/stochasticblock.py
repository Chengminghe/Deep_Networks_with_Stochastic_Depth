from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from numpy.random import choice
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from utils.resnetblock import TwoLayerBlock, ThreeLayerBlock


class TwoLayerStochastic(TwoLayerBlock):
    def __init__(self,kernel_size,filters,keep_prob,down_sampling=False):
        super(TwoLayerStochastic,self).__init__(kernel_size,filters,down_sampling)
        self.keep_prob = keep_prob

        
    def call(self,x,training=False):
        if training == True:
            if choice(2,p=[1-self.keep_prob,self.keep_prob]) == 1:
                self.conv_1.trainable = True
                self.bn_1.trainable = True
                self.conv_2.trainable = True
                self.bn_2.trainable = True  
                return super().call(x,training=training)
            else:
                self.conv_1.trainable = False
                self.bn_1.trainable = False
                self.conv_2.trainable = False
                self.bn_2.trainable = False
                if self.down_sampling:
                    x = self.conv_3(x)
                    x = self.bn_3(x,training=training)               
                return Activation(relu)(x)
        else:
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
            x = self.add([self.keep_prob*x,x_prev])
            return self.ac_3(x)

class ThreeLayerStochastic(ThreeLayerBlock):
    def __init__(self,filters,keep_prob,down_sampling=False):
        super(ThreeLayerStochastic,self).__init__(filters,down_sampling)
        self.keep_prob = keep_prob


    def call(self,x,training=False):
        if training == True:
            if choice(2,p=[1-self.keep_prob,self.keep_prob]) == 1:
                self.conv_1.trainable = True
                self.bn_1.trainable = True
                self.conv_2.trainable = True
                self.bn_2.trainable = True 
                self.conv_3.trainable = True
                self.bn_3.trainable = True
                return super().call(x,training=training)
            else:
                self.conv_1.trainable = False
                self.bn_1.trainable = False
                self.conv_2.trainable = False
                self.bn_2.trainable = False
                self.conv_3.trainable = False
                self.bn_3.trainable = False
                if self.down_sampling:
                    x = self.conv_4(x)
                    x = self.bn_4(x,training=training)               
                return Activation(relu)(x)
        else:
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
            if self.down_sampling:
                x_prev = self.conv_4(x_prev)
                x_prev = self.bn_4(x_prev,training=training)
            x = self.add([self.keep_prob*x,x_prev])
            return self.ac_4(x)    

    


