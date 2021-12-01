import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D,BatchNormalization, Dropout
from tensorflow.keras import Model

class ResNet(Model):

    def __init__(self, num_block, input_shape, output_size=100):
        '''
        For training the cifar10/cifar100 data in the ResNet paper
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(ResNet, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        self.num_block = num_block
        self.layers = dict()
        self.layers['conv1'] = Conv2D(filters=16,kernel_size=(3, 3),input_shape=input_shape,activation='relu'
                                ,kernel_initializer=tf.keras.initializers.HeNormal(),padding="same",use_bias=True)

        for i in range(2*n):
            self.layers['conv1_%d'%(i+1)] = Conv2D(filters=16,kernel_size=(3, 3),activation='relu',padding="same"
                                    ,use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())

        self.layers['conv2_1'] = Conv2D(filters=32,kernel_size=(3, 3),activation='relu',strides=(2,2),padding="same"
             ,use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        for i in range(2*n-1):
            self.layers['conv2_%d'%(i+2)] = Conv2D(filters=32,kernel_size=(3, 3),activation='relu',padding="same"
                     ,use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        
        self.layers['conv3_1'] = Conv2D(filters=64,kernel_size=(3, 3),activation='relu',strides=(2,2),padding="same"
             ,use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())
        for i in range(2*n-1):
            self.layers['conv3_%d'%(i+2)] = Conv2D(filters=64,kernel_size=(3, 3),activation='relu',padding="same"
                     ,use_bias=True,kernel_initializer=tf.keras.initializers.HeNormal())

        
        self.layers['avgpool'] = GlobalAveragePooling2D()
        self.layers['bn'] = BatchNormalization()
        self.layers['fc'] = Dense(10,activation='softmax')
    
        #############################################################
        #                          END TODO                         #                                              
        #############################################################

    
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        num_block = self.num_block
        x = self.layers['conv1'](x)
        x_prev = x
        
        for i in range(2*num_block):
            if x%2 == 0:
                x = layers['conv1_%d'%(i+1)](x)
            else:
                x = layers['conv1_%d'%(i+1)](x) + x_prev
                x_prev = x
        


        for i in range(2*num_block):
            if x%2 == 0:
                x = layers['conv2_%d'%(i+1)](x)
            else:
                if x.shape == x_prev.shape:
                    x = layers['conv2_%d'%(i+1)](x) + x_prev
                    x_prev = x
                else:
                    x_prev = Conv2D(filters=32,kernel_size=(1, 1),strides=(2, 2)
                            ,use_bias=False,kernel_initializer=tf.keras.initializers.Ones())(x_prev)/16
                    x = layers['conv2_%d'%(i+1)](x) + x_prev
                    x_prev = x

        for i in range(2*num_block):
            if x%2 == 0:
                x = layers['conv3_%d'%(i+1)](x)
            else:
                if x.shape == x_prev.shape:
                    x = layers['conv3_%d'%(i+1)](x) + x_prev
                    x_prev = x
                else:
                    x_prev = Conv2D(filters=64,kernel_size=(1, 1),strides=(2, 2)
                            ,use_bias=False,kernel_initializer=tf.keras.initializers.Ones())(x_prev)/32
                    x = layers['conv3_%d'%(i+1)](x) + x_prev
                    x_prev = x

        x = self.layers['avgpool'](x)
        x = self.layers['bn'](x)
        out = self.layers['fc'](x)
        
        return out
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        

