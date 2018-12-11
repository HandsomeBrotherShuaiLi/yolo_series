"""
written by: Li Shuai
Time: 2018/12/04
Localization: NIO DLLab, Shanghai
darknet53 model body for self-driving experiments
"""
from keras.models import Model
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,Dense
from keras.layers import add,Activation,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

def conv2d_unit(x,filters,kernels,strides=1):
    """
    This function defines a 2D convolution operation
    with BN and LeakyReLU
    :param x: input tensor
    :param filters:
    :param kernels:
    :param strides:
    :return:
    """
    x=Conv2D(filters,kernels,padding='same',
             strides=strides,activation='linear',
             kernel_regularizer=l2(5e-4))(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(inputs,filters):
    """
    2D convolutional operation with BN and LeakyReLU
    :param inputs:
    :param filters:
    :return:
    """
    x=conv2d_unit(inputs,filters,(1,1))
    x=conv2d_unit(x,2*filters,(3,3))
    x=add([inputs,x])
    x=Activation('linear')(x)
    return x

def stack_residual_block(inputs,filters,n):
    """
    stacked residual block
    :param inputs:
    :param filters:
    :param n:
    :return:
    """
    x=residual_block(inputs,filters)
    for i in range(n-1):
        x=residual_block(x,filters)
    return x

def darknet(input_shape,number_class=2):
    """
    darknet-53 model， we need resize the input shape of custom dataset
    :return:
    """
    inputs=Input(shape=input_shape)

    x=conv2d_unit(inputs,32,(3,3))

    x=conv2d_unit(x,64,(3,3),strides=2)
    x=stack_residual_block(x,32,n=1)

    x=conv2d_unit(x,128,(3,3),strides=2)
    x=stack_residual_block(x,64,n=2)

    x=conv2d_unit(x,256,(3,3),strides=2)
    x=stack_residual_block(x,128,n=8)

    x=conv2d_unit(x,512,(3,3),strides=2)
    x=stack_residual_block(x,256,n=8)

    x=conv2d_unit(x,1024,(3,3),strides=2)
    x=stack_residual_block(x,512,n=4)

    x=GlobalAveragePooling2D()(x)

    # the number of original classification label is 1000 (Imagenet)
    # here we can modify as we need
    x=Dense(number_class,activation='softmax')(x)

    model=Model(inputs,x)

    return model

if __name__=="__main__":
    model=darknet(input_shape=(5184,3888,3),number_class=20)
    print(model.summary())
