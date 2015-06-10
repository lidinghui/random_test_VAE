# training maxout on mnist
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Softmax
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
from blocks.bricks.cost import MisclassificationRate
from blocks.algorithms import Momentum, RMSProp, Scale

floatX = theano.config.floatX
from fuel.datasets import MNIST
from maxout_classifier import Maxout
from contextlib import closing

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

def maxout_mnist_test():

    maxout = Maxout()
    return
    x = T.tensor4('features')
    y = T.imatrix('targets')
    batch_size = 128
    predict = maxout.apply(x)

    cost = Softmax().categorical_cross_entropy(y.flatten(), predict)
    cost.name = 'cost'
    cg = ComputationGraph(cost)
    """
    weights = VariableFilter(roles=[WEIGHT])(cg_.variables)

    cost = cost + 0.001*sum([T.sum(p**2) for p in weights])
    cost.name = 'cost'
    cg = ComputationGraph(cost)
    """
    temp = cg.parameters
    for t, i in zip(temp, range(len(temp))):
        t.name = t.name+str(i)+"maxout"


    """
    error_rate = errors(predict, y)
    error_rate.name = 'error'
    """
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

    # save here
    from blocks.serialization import dump
    with closing(open('../data_mnist/maxout', 'w')) as f:
	    dump(vae, f)
    


if __name__ == '__main__':
    maxout_mnist_test()
