# maxout classifier
import numpy as np
import theano
import theano.tensor as T
from blocks.initialization import Constant, Uniform
from blocks.bricks.base import application

#from batch_normalize import ConvolutionalLayer, ConvolutionalActivation, Linear
# change for cpu tests
from convolution_stride import ConvolutionalLayer, Maxout_
from blocks.bricks import Linear
floatX = theano.config.floatX
from blocks.bricks.conv import Flattener
from blocks.bricks import Initializable


class Maxout(Initializable):
    
    def __init__(self, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        filter_size = (8, 8)
        activation = Maxout_(num_pieces=2).apply
        pooling_size = 4
        pooling_step = 2
        pad = 0
        image_size = (28, 28)
        num_channels = 1
        num_filters = 48
        layer0 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=(pooling_step, pooling_step),
                                pad=pad,
                                image_size=image_size,
                                num_channels=num_channels,
                                weights_init=Uniform(width=0.01),
                                biases_init=Uniform(width=0.01),
                                name="layer_0_maxout")
        layer0.initialize()

        num_filters = 48
        filter_size = (8,8)
        pooling_size = 4
        pooling_step = 2
        pad = 3  
        image_size = (layer0.get_dim('output')[1],
                  layer0.get_dim('output')[2])
        num_channels = layer0.get_dim('output')[0]
        layer1 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=(pooling_step, pooling_step),
                                pad=pad,
                                image_size=image_size,
                                num_channels=num_channels,
                                weights_init=Uniform(width=0.01),
                                biases_init=Uniform(width=0.01),
                                name="layer_1_maxout")
        layer1.initialize()

        num_filters = 24
        filter_size=(5,5)
        pooling_size = 2
        pooling_step = 2
        pad = 3
        activation = Maxout_(num_pieces=4).apply
        image_size = (layer1.get_dim('output')[1],
                  layer1.get_dim('output')[2])
        num_channels = layer1.get_dim('output')[0]
        layer2 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=(pooling_step, pooling_step),
                                pad = pad,
                                image_size=image_size,
                                num_channels=num_channels,
                                weights_init=Uniform(width=0.01),
                                biases_init=Uniform(width=0.01),
                                name="layer_2_maxout")
        layer2.initialize()

        self.conv_layers = [layer0, layer1, layer2]


        self.mlp_layer = Linear(54, 10, 
                        weights_init=Uniform(width=0.01),
                        biases_init=Uniform(width=0.01), name="layer_5_maxout")
        self.mlp_layer.initialize()

    @application(inputs=['input_'], outputs=['output_'])
    def apply(self, input_):
        output_conv = input_
        for layer in self.conv_layers:
            output_conv = layer.apply(output_conv)
        output_conv = Flattener().apply(output_conv)
        output_ = self.mlp_layer.apply(output_conv)
        return output_
