# build communation model
from blocks.serialization import load, dump
import theano
import theano.tensor as T
import numpy as np
from vae import Qsampler, VAEModel
from samples_save import ImagesSamplesSave
from blocks.initialization import Constant, NdarrayInitialization, Sparse, Orthogonal
from blocks.bricks import MLP, Softmax, Rectifier
from blocks.bricks.cost import MisclassificationRate
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


def test_communication(path_vae_svhn, path_vae_mnist,
                       path_maxout_mnist):
                       
    # load models
    vae_svhn = load(path_vae_svhn)
    vae_mnist = load(path_vae_mnist)
    # write an object maxout
    classifier = load(path_maxout_mnist)
    
    # vae whose prior is a zero mean unit variance normal distribution
    activation = Rectifier()
    full_weights_init = Orthogonal()
    weights_init = full_weights_init
    
    layers = [vae_svhn.sampler.get_dim('output'), 200, 200, 200, vae_mnist.sampler.get_dim('input')]
    encoder_layers = layers[:-1]
    encoder_mlp = MLP([activation] * (len(encoder_layers)-1),
              encoder_layers,
              name="MLP_connection", biases_init=Constant(0.), weights_init=weights_init)

    enc_dim = encoder_layers[-1]
    z_dim = layers[-1]
    #sampler = Qlinear(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    decoder_layers = layers[:]  ## includes z_dim as first layer
    decoder_layers.reverse()
    decoder_mlp = MLP([activation] * (len(decoder_layers)-2) + [Rectifier()],
              decoder_layers,
              name="MLP_connection", biases_init=Constant(0.), weights_init=weights_init)

    
    vae = VAEModel(encoder_mlp, sampler, decoder_mlp)
    vae.initialize()
    
    # do the connection
    
    x = T.matrix('x') # SVHN samples preprocessed with local contrast normalization
    y = T.imatrix('y')
    
    svhn_z = vae_svhn.encoder_mlp.apply(x)
    mnist_decode = vae_mnist.decoder_mlp.apply(svhn_z)
    prediction = classifier.apply(mnist_decode)
    
    cost = Softmax().categorical_cross_entropy(y.flatten(), prediction)
    cost.name = 'cost'
    cg = ComputationGraph(cost) # probably discard some of the parameters
    error_brick = MisclassificationRate()
    error_rate = error_brick.apply(y.flatten(), prediction)
    
    # training here

if __name__ == '__main__':
    test_communication()