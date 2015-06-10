# training a maxout on mnist reconstructed by a VAE
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Softmax
from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph

from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
#from batch_normalize import ConvolutionalLayer, ConvolutionalActivation, Linear
# change for cpu tests
from blocks.bricks.cost import MisclassificationRate
from blocks.algorithms import Momentum, RMSProp, Scale

floatX = theano.config.floatX
from fuel.datasets import MNIST
from maxout_classifier import Maxout
from contextlib import closing
from blocks.serialization import load
from fuel.transformers import Flatten


def maxout_vae_mnist_test(path_vae_mnist):

    # load vae model on mnist
    vae_mnist = load(path_vae_mnist)
    maxout = Maxout()
    x = T.matrix('features')
    y = T.imatrix('targets')
    batch_size = 128
    z, _ = vae_mnist.sampler.sample(vae_mnist.encoder_mlp.apply(x))
    predict = maxout.apply(z)

    cost = Softmax().categorical_cross_entropy(y.flatten(), predict)
    y_hat = Softmax().apply(predict)
    cost.name = 'cost'
    cg = ComputationGraph(cost)

    temp = cg.parameters
    for t, i in zip(temp, range(len(temp))):
        t.name = t.name+str(i)+"maxout"

    error_brick = MisclassificationRate()
    error_rate = error_brick.apply(y, y_hat) 

    # training
    step_rule = RMSProp(0.01, 0.9)
    #step_rule = Momentum(0.2, 0.9)
    train_set = MNIST('train')
    test_set = MNIST("test")

    data_stream_train = Flatten(DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size)))

    data_stream_test =Flatten(DataStream.default_stream(
            test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size)))

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost], data_stream=data_stream_train, prefix="train")
    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="test")


    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_epochs=50),
                    Printing(every_n_epochs=1)
                  ]

    main_loop = MainLoop(data_stream=data_stream_train,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()

    # save here
    from blocks.serialization import dump
    with closing(open('../data_mnist/maxout', 'w')) as f:
	    dump(maxout, f)
    


if __name__ == '__main__':
    maxout_vae_mnist_test()
