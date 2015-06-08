import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Softmax, Linear
from blocks.initialization import Constant, Uniform
from blocks.algorithms import GradientDescent
from blocks.roles import WEIGHT, BIAS
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter

from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
import operator
from blocks.roles import PARAMETER
#from batch_normalize import ConvolutionalLayer, ConvolutionalActivation, Linear
# change for cpu tests
from blocks.bricks.conv import ConvolutionalSequence
from convolution_stride import ConvolutionalLayer, Maxout_
from blocks.bricks.conv import Flattener
from blocks.bricks import Linear
from blocks.bricks.cost import MisclassificationRate
from blocks.algorithms import Momentum, RMSProp, Scale
#from fuel.datasets.hdf5 import H5PYDataset
#from fuel.transformers import Flatten

floatX = theano.config.floatX
#import h5py
#from contextlib import closing
#from momentum import Momentum_dict
#import re
from maxout_extension import Clip_param
from fuel.datasets import MNIST

def errors(p_y_given_x, y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """
    y = T.cast(y, 'int8')
    y_pred = T.argmax(p_y_given_x, axis=1)
    y_pred = y_pred.dimshuffle((0, 'x'))
    y_pred = T.cast(y_pred, 'int8')
    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y), dtype=floatX)
    else:
        raise NotImplementedError()

def init_param(params, name, value):

    if name in params:
        param_i = params[name]
        shape = param_i.get_value().shape
        #print (name, shape, value.shape)
        param_i.set_value((value.reshape(shape)).astype(floatX))
    else:
        raise Exception("unknown parameter")

def build_params(cnn_layer, mlp_layer):

    params = []
    names = []
    input = T.tensor4()
    x = T.matrix()

    for i, layer in zip(range(len(cnn_layer)), cnn_layer):
        param_layer = VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(input).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i)+"_"+p.name
            names.append(p.name)
            params.append(p)   

    for i, layer in zip(range(len(mlp_layer)), mlp_layer):
        param_layer= VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i+len(cnn_layer))+"_"+p.name
            names.append(p.name)
            params.append(p)

    return params, names


def maxout_mnist_test():
    # if it is working
    # do a class
    x = T.tensor4('features')
    y = T.imatrix('targets')
    batch_size = 128
    # maxout convolutional layers
    # layer0
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
                                name="layer_0")
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
                                name="layer_1")
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
                                name="layer_2")
    layer2.initialize()

    conv_layers = [layer0, layer1, layer2]
    output_conv = x
    for layer in conv_layers :
        output_conv = layer.apply(output_conv)
    output_conv = Flattener().apply(output_conv)

    mlp_layer = Linear(54, 10, 
                        weights_init=Uniform(width=0.01),
                        biases_init=Uniform(width=0.01), name="layer_5")
    mlp_layer.initialize()

    output_mlp = mlp_layer.apply(output_conv)

    params, names = build_params(conv_layers, [mlp_layer])

    cost = Softmax().categorical_cross_entropy(y.flatten(), output_mlp)
    cost.name = 'cost'
    cg_ = ComputationGraph(cost)
    weights = VariableFilter(roles=[WEIGHT])(cg_.variables)
    cost = cost + 0.001*sum([sum(p**2) for p in weights])
    cg = ComputationGraph(cost)
    error_rate = errors(output_mlp, y)
    error_rate.name = 'error'

    # training
    step_rule = RMSProp(0.01, 0.9)
    #step_rule = Momentum(0.2, 0.9)
    train_set = MNIST('train')
    test_set = MNIST("test")

    data_stream = DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))

    data_stream_monitoring = DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))

    data_stream_test =DataStream.default_stream(
            test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size))

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    monitor_train = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_monitoring, prefix="train")
    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="test")


    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_epochs=50),
                    Printing(every_n_epochs=1)
                  ]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()


if __name__ == '__main__':
    maxout_mnist_test()
