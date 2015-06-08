# test vae on SVHN
import theano
import theano.tensor as T
import numpy as np
from vae import Qsampler, VAEModel
from samples_save import ImagesSamplesSave
from blocks.initialization import Constant, NdarrayInitialization, Sparse, Orthogonal
from blocks.bricks import MLP, Tanh, WEIGHT, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
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

floatX = theano.config.floatX
from fuel.datasets import MNIST

def test_vae():

    activation = Rectifier()
    full_weights_init = Orthogonal()
    weights_init = full_weights_init
    
    layers = [3*32*32, 500, 100, 100, 80]
    encoder_layers = layers[:-1]
    encoder_mlp = MLP([activation] * (len(encoder_layers)-1),
              encoder_layers,
              name="MLP_enc", biases_init=Constant(0.), weights_init=weights_init)

    enc_dim = encoder_layers[-1]
    z_dim = layers[-1]
    #sampler = Qlinear(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    decoder_layers = layers[:]  ## includes z_dim as first layer
    decoder_layers.reverse()
    decoder_mlp = MLP([activation] * (len(decoder_layers)-2) + [Rectifier()],
              decoder_layers,
              name="MLP_dec", biases_init=Constant(0.), weights_init=weights_init)

    
    vae = VAEModel(encoder_mlp, sampler, decoder_mlp)
    vae.initialize()

    x = T.matrix('features')
    batch_size = 128
    x_recons, kl_terms = vae.reconstruct(x)
    recons_term = BinaryCrossEntropy().apply(x, T.clip(x_recons, 1e-4, 1 - 1e-4))
    recons_term.name = "recons_term"

    cost = recons_term + kl_terms.mean()
    cost.name = "cost"
    cg = ComputationGraph([cost])

    step_rule = Momentum(0.001, 0.2)

    dataset_path="/Tmp/ducoffem/SVHN/all.h5"
    train_set = H5PYDataset(os.path.join(dataset_hdf5_file, "all.h5"), which_set='train')
    test_set = H5PYDataset(os.path.join(dataset_hdf5_file, "all.h5"), which_set='valid')
    
    data_stream = DataStream.default_stream(
        train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
        
    data_stream_test = DataStream.default_stream(
        test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size))


    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost], prefix="train")
    monitor_valid = DataStreamMonitoring(
        variables=[cost], data_stream=data_stream_test, prefix="valid")

    drawing_samples = ImagesSamplesSave("./data_svhn", vae, (3, 32, 32), every_n_epochs=1)
    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_epochs=400),
                    Printing(every_n_epochs=10),
                    drawing_samples,
                    Dump("./data_svhn/model_svhn")
                  ]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()

if __name__ == '__main__':
    test_vae()