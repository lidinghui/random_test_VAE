# build communation model
from blocks.serialization import load, dump
import theano
import theano.tensor as T
import numpy as np
from vae import Qsampler, VAEModel
from samples_save import ImagesSamplesSave
from blocks.initialization import Constant, NdarrayInitialization, Sparse, Orthogonal
from blocks.bricks import MLP, Softmax, Rectifier
from blocks.bricks.cost import MisclassificationRate, BinaryCrossEntropy
from blocks.graph import ComputationGraph

from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.algorithms import Momentum, RMSProp, Scale, Adam
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent
from maxout_classifier import Maxout
from fuel.datasets.hdf5 import H5PYDataset
import os
import re

def test_communication(path_vae_mnist,
                       path_maxout_mnist):
                       
    # load models
    vae_mnist = load(path_vae_mnist)
    # get params : to be remove from the computation graph

    # write an object maxout
    classifier = Maxout()
    # get params : to be removed from the computation graph

    # vae whose prior is a zero mean unit variance normal distribution
    activation = Rectifier()
    full_weights_init = Orthogonal()
    weights_init = full_weights_init

    # SVHN en niveau de gris
    layers = [32*32, 200, 200, 200, 50]
    encoder_layers = layers[:-1]
    encoder_mlp = MLP([activation] * (len(encoder_layers)-1),
              encoder_layers,
              name="MLP_SVHN_encode", biases_init=Constant(0.), weights_init=weights_init)

    enc_dim = encoder_layers[-1]
    z_dim = layers[-1]
    sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    decoder_layers = layers[:]  ## includes z_dim as first layer
    decoder_layers.reverse()
    decoder_mlp = MLP([activation] * (len(decoder_layers)-2) + [Rectifier()],
              decoder_layers,
              name="MLP_SVHN_decode", biases_init=Constant(0.), weights_init=weights_init)

    
    vae_svhn = VAEModel(encoder_mlp, sampler, decoder_mlp)
    vae_svhn.initialize()

    # do the connection
    
    x = T.tensor4('x') # SVHN samples preprocessed with local contrast normalization
    x_ = (T.sum(x, axis=1)).flatten(ndim=2)
    y = T.imatrix('y')
    batch_size = 512

    svhn_z, _ = vae_svhn.sampler.sample(vae_svhn.encoder_mlp.apply(x_))
    mnist_decode = vae_mnist.decoder_mlp.apply(svhn_z)
    # reshape
    shape = mnist_decode.shape
    mnist_decode = mnist_decode.reshape((shape[0], 1, 28, 28))
    prediction = classifier.apply(mnist_decode)
    y_hat = Softmax().apply(prediction)

    x_recons, kl_terms = vae_svhn.reconstruct(x_)
    recons_term = BinaryCrossEntropy().apply(x_, T.clip(x_recons, 1e-4, 1 - 1e-4))
    recons_term.name = "recons_term"

    cost_A = recons_term + kl_terms.mean()
    cost_A.name = "cost_A"

    cost_B = Softmax().categorical_cross_entropy(y.flatten(), prediction)
    cost_B.name = 'cost_B'

    cost = cost_B
    cost.name = "cost"
    cg = ComputationGraph(cost) # probably discard some of the parameters
    parameters = cg.parameters
    params = []
    for t in parameters:
        if not re.match(".*mnist", t.name):
            params.append(t)

    """
    f = theano.function([x], cost_A)
    value_x = np.random.ranf((1, 3, 32, 32)).astype("float32")
    print f(value_x)
    
    return
    """
    error_brick = MisclassificationRate()
    error_rate = error_brick.apply(y.flatten(), y_hat)
    error_rate.name = "error_rate"
    
    # training here
    step_rule = RMSProp(0.001,0.99)

    dataset_hdf5_file="/Tmp/ducoffem/SVHN/"
    train_set = H5PYDataset(os.path.join(dataset_hdf5_file, "all.h5"), which_set='train')
    test_set = H5PYDataset(os.path.join(dataset_hdf5_file, "all.h5"), which_set='valid')
    
    data_stream = DataStream.default_stream(
        train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
        
    data_stream_test = DataStream.default_stream(
        test_set, iteration_scheme=SequentialScheme(2000, batch_size))


    algorithm = GradientDescent(cost=cost, params=params,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost], prefix="train", every_n_batches=10)
    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="valid", every_n_batches=10)

    # drawing_samples = ImagesSamplesSave("../data_svhn", vae, (3, 32, 32), every_n_epochs=1)
    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_batches=10000),
                    Printing(every_n_batches=10)
                  ]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()

if __name__ == '__main__':
    path_vae_mnist = "../data_mnist/model"
    path_maxout_mnist =  "../data_mnist/maxout.zip"
    test_communication(path_vae_mnist,
                       path_maxout_mnist)
