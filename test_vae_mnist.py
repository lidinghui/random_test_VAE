# test vae on mnist
import theano
import theano.tensor as T
import numpy as np
from vae import Qsampler, VAEModel
from samples_save import ImagesSamplesSave
from blocks.initialization import Constant, NdarrayInitialization, Sparse, Orthogonal
from blocks.bricks import MLP, Tanh, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
from blocks.graph import ComputationGraph

from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.algorithms import Momentum, RMSProp, Scale
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent
from contextlib import closing
floatX = theano.config.floatX
from fuel.datasets import MNIST

def test_vae():

    activation = Rectifier()
    full_weights_init = Orthogonal()
    weights_init = full_weights_init
    
    layers = [784, 800, 800, 800, 50]
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
    batch_size = 512
    x_recons, kl_terms = vae.reconstruct(x)
    recons_term = BinaryCrossEntropy().apply(x, T.clip(x_recons, 1e-4, 1 - 1e-4))
    recons_term.name = "recons_term"

    cost = recons_term + kl_terms.mean()
    cost.name = "cost"
    cg = ComputationGraph(cost)
    temp = cg.parameters
    for t, i in zip(temp, range(len(temp))):
        t.name = t.name+str(i)+"vae_mnist"


    step_rule = RMSProp(0.0001, 0.95)

    train_set = MNIST('train')

    train_set.sources = ("features", )
    test_set = MNIST("test")
    test_set.sources = ("features", )

    data_stream = Flatten(DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size)))

    data_stream_monitoring = Flatten(DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size)))

    data_stream_test = Flatten(DataStream.default_stream(
            test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size)))

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost], prefix="train", every_n_epochs=1)
    monitor_valid = DataStreamMonitoring(
        variables=[cost], data_stream=data_stream_test, prefix="valid", every_n_epochs=1)

    # drawing_samples = ImagesSamplesSave("../data_mnist", vae, (28, 28), every_n_epochs=1)
    extensions = [  monitor_train,
                    monitor_valid,
                    FinishAfter(after_n_epochs=400),
                    Printing(every_n_epochs=1)
                  ]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()

    from blocks.serialization import dump
    with closing(open('../data_mnist/model_0', 'w')) as f:
	    dump(vae, f)

if __name__ == '__main__':
    test_vae()

"""
def main(name, model, epochs, batch_size, learning_rate, bokeh, layers, gamma,
         rectifier, predict, dropout, qlinear, sparse):
    runname = "vae%s-L%s%s%s%s-l%s-g%s-b%d" % (name, layers,
                                            'r' if rectifier else '',
                                            'd' if dropout else '',
                                            'l' if qlinear else '',
                                      shnum(learning_rate), shnum(gamma), batch_size//100)
    if rectifier:
        activation = Rectifier()
        full_weights_init = Orthogonal()
    else:
        activation = Tanh()
        full_weights_init = Orthogonal()

    if sparse:
        runname += '-s%d'%sparse
        weights_init = Sparse(num_init=sparse, weights_init=full_weights_init)
    else:
        weights_init = full_weights_init

    layers = map(int,layers.split(','))

    encoder_layers = layers[:-1]
    encoder_mlp = MLP([activation] * (len(encoder_layers)-1),
              encoder_layers,
              name="MLP_enc", biases_init=Constant(0.), weights_init=weights_init)

    enc_dim = encoder_layers[-1]
    z_dim = layers[-1]
    if qlinear:
        sampler = Qlinear(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)
    else:
        sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, biases_init=Constant(0.), weights_init=full_weights_init)

    decoder_layers = layers[:]  ## includes z_dim as first layer
    decoder_layers.reverse()
    decoder_mlp = MLP([activation] * (len(decoder_layers)-2) + [Sigmoid()],
              decoder_layers,
              name="MLP_dec", biases_init=Constant(0.), weights_init=weights_init)


    vae = VAEModel(encoder_mlp, sampler, decoder_mlp)
    vae.initialize()

    x = tensor.matrix('features')

    if predict:
        mean_z, enc = vae.mean_z(x)
        # cg = ComputationGraph([mean_z, enc])
        newmodel = Model([mean_z,enc])
    else:
        x_recons, kl_terms = vae.reconstruct(x)
        recons_term = BinaryCrossEntropy().apply(x, x_recons)
        recons_term.name = "recons_term"

        cost = recons_term + kl_terms.mean()
        cg = ComputationGraph([cost])

        if gamma > 0:
            weights = VariableFilter(roles=[WEIGHT])(cg.variables)
            cost += gamma * blocks.theano_expressions.l2_norm(weights)

        cost.name = "nll_bound"
        newmodel = Model(cost)

        if dropout:
            weights = [v for k,v in newmodel.get_params().iteritems()
                       if k.find('MLP')>=0 and k.endswith('.W') and not k.endswith('MLP_enc/linear_0.W')]
            cg = apply_dropout(cg,weights,0.5)
            target_cost = cg.outputs[0]
        else:
            target_cost = cost

    if name == 'mnist':
        if predict:
            train_ds = MNIST("train")
        else:
            train_ds = MNIST("train", sources=['features'])
        test_ds = MNIST("test")
    else:
        datasource_dir = os.path.join(fuel.config.data_path, name)
        datasource_fname = os.path.join(datasource_dir , name+'.hdf5')
        if predict:
            train_ds = H5PYDataset(datasource_fname, which_set='train')
        else:
            train_ds = H5PYDataset(datasource_fname, which_set='train', sources=['features'])
        test_ds = H5PYDataset(datasource_fname, which_set='test')
    train_s = DataStream(train_ds,
                 iteration_scheme=SequentialScheme(
                     train_ds.num_examples, batch_size))
    test_s = DataStream(test_ds,
                 iteration_scheme=SequentialScheme(
                     test_ds.num_examples, batch_size))

    if predict:
        from itertools import chain
        fprop = newmodel.get_theano_function()
        allpdata = None
        alledata = None
        f = train_s.sources.index('features')
        assert f == test_s.sources.index('features')
        sources = test_s.sources
        alllabels = dict((s,[]) for s in sources if s != 'features')
        for data in chain(train_s.get_epoch_iterator(), test_s.get_epoch_iterator()):
            for s,d in zip(sources,data):
                if s != 'features':
                    alllabels[s].extend(list(d))

            pdata, edata = fprop(data[f])
            if allpdata is None:
                allpdata = pdata
            else:
                allpdata = np.vstack((allpdata, pdata))
            if alledata is None:
                alledata = edata
            else:
                alledata = np.vstack((alledata, edata))
        print 'Saving',allpdata.shape,'intermidiate layer, for all training and test examples, to',name+'_z.npy'
        np.save(name+'_z', allpdata)
        print 'Saving',alledata.shape,'last encoder layer to',name+'_e.npy'
        np.save(name+'_e', alledata)
        print 'Saving additional labels/targets:',','.join(alllabels.keys()),
        print ' of size',','.join(map(lambda x: str(len(x)),alllabels.values())),
        print 'to',name+'_labels.pkl'
        with open(name+'_labels.pkl','wb') as fp:
            pickle.dump(alllabels, fp, -1)
    else:
        algorithm = GradientDescent(
            cost=target_cost, params=cg.parameters,
            step_rule=Adam(learning_rate)  # Scale(learning_rate=learning_rate)
        )
        extensions = []
        if model:
            extensions.append(LoadFromDump(model))

        extensions += [Timing(),
                      FinishAfter(after_n_epochs=epochs),
                      DataStreamMonitoring(
                          [recons_term, cost],
                          test_s,
                          prefix="test"),
                      TrainingDataMonitoring(
                          [cost,
                           aggregation.mean(algorithm.total_gradient_norm)],
                          prefix="train",
                          after_epoch=True),
                      Dump(runname, every_n_epochs=10),
                      Printing()]

        if bokeh:
            extensions.append(Plot(
                'Auto',
                channels=[
                    ['test_recons_term','test_nll_bound','train_nll_bound'
                     ],
                    ['train_total_gradient_norm']]))

        main_loop = MainLoop(
            algorithm,
            train_s,
            model=newmodel,
            extensions=extensions)

        main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a Variational-Autoencoder.")
    parser.add_argument("--name", default="mnist",
                        help="name of hdf5 data set")
    parser.add_argument("--model",
                        help="start model to read")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=500, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--bokeh", action='store_true', default=False,
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--layers",
                default="784,100,20", help="number of units in each layer of the encoder"
                                           " (use 784, on first layer, for mnist.)"
                                           " The last number (e.g. 20) is the dimension of the intermidiate layer."
                                           " The decoder has the same layers as the encoder but in reverse"
                                           " (e.g. 100, 784)")
    parser.add_argument("--gamma", type=float,
                default=3e-4, help="L2 weight")
    parser.add_argument("-r","--rectifier",action='store_true',default=False,
                        help="Use RELU activation on hidden (default Tanh)")
    parser.add_argument("-p","--predict",action='store_true',default=False,
                        help="Generate prediction of the  intermidate layer and last layer of the encoder"
                             " instead of training."
                             " You must supply a pre-trained model and define all parameters to be the same"
                             " as in training. ")
    parser.add_argument("-d","--dropout",action='store_true',default=False,
                        help="Use dropout")
    parser.add_argument("-l","--qlinear",action='store_true',default=False,
                        help="Perform a deterministic linear transformation instead of sampling"
                             " on the intermidiate layer")
    parser.add_argument("-s","--sparse",type=int,
                        help="Use sparse weight initialization. Give the number of non zero weights")
    args = parser.parse_args()
    main(**vars(args))
""" 
