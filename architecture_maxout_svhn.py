"""
an implement of maxout for SVHN
on blocks
"""

import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Rectifier, Softmax, MLP, Identity
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.algorithms import GradientDescent
from blocks.roles import WEIGHT, BIAS, INPUT
from blocks.graph import ComputationGraph, apply_dropout
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
from convolution_stride import ConvolutionalLayer
from blocks.bricks.conv import MaxPooling, Flattener
from blocks.bricks import Linear

from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.algorithms import Momentum, RMSProp
from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Flatten

floatX = theano.config.floatX
import h5py
from contextlib import closing
from momentum import Momentum_dict

def init_param(params, name, value):

    if name in params:
        param_i = params[name]
        shape = param_i.get_value().shape
        #print (name, shape, value.shape)
        param_i.set_value((value.reshape(shape)).astype(floatX))
    else:
        raise Exception("unknown parameter")

def return_param(params, name):

    if name in params:
        return params[name].get_value()
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
            print (p.name, p.get_value().shape)
        # do the same for the input -> easier for the second level of dropout
        param_layer = VariableFilter(roles=[INPUT])(ComputationGraph(layer.apply(input).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i)+"_input"     

    for i, layer in zip(range(len(mlp_layer)), mlp_layer):
        param_layer= VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i+len(cnn_layer))+"_"+p.name
            names.append(p.name)
            params.append(p)
            print (p.name, p.get_value().shape)
        param_layer= VariableFilter(roles=[INPUT])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i+len(cnn_layer))+"_input"

    return params, names


def build_architecture( step_flavor,
                        drop_conv, drop_mlp,
                        L_nbr_filters, L_nbr_hidden_units,
                        dataset_hdf5_file=None):

    # dataset_hdf5_file is like "/rap/jvb-000-aa/data/ImageNet_ILSVRC2010/pylearn2_h5/imagenet_2010_train.h5"

    assert len(L_nbr_filters) == 4
    assert len(L_nbr_hidden_units) == 4
    # Note that L_nb_hidden_units[0] is something constrained
    # by the junction between the filters and the fully-connected section.
    # Find this value in the JSON file that describes the model.

    x=T.tensor4('x')
    
    # TODO : check the pre processing step of maxout !!!!
    if dataset_hdf5_file is not None:
        with closing(h5py.File(dataset_hdf5_file, 'r')) as f:
            x_mean = (f['x_mean']).value
            x_mean = x_mean.reshape((1, 3, 32, 32)) # equivalent to a dimshuffle on the number of elem in the batch
        x = (x - x_mean)
        # TODO : maybe normalize ?
    
    y = T.imatrix('y')

    # Convolution Maxout Layer
    filter_size = (5, 5)
    activation = Maxout(num_pieces=2).apply
    pooling_size = 3
    pooling_step = 2
    pad = 4
    num_filters = L_nbr_filters[0] - int(drop_conv[0]*L_nbr_filters[0])
    layer0 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=pooling_step,
                                name="layer0")

    # set of hyperparameters
    # padding

    num_filters = L_nbr_filters[1] - int(drop_conv[1]*L_nbr_filters[1])
    filter_size = (5,5)
    pooling_size = 3
    pooling_step = 2
    pad = 3
    layer1 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=pooling_step,
                                pad=pad
                                name="layer1")

    num_channels = num_filters
    num_filters = L_nbr_filters[2] - int(drop_conv[2]*L_nbr_filters[2])
    filter_size=(5,5)
    pooling_size = 2
    pad = 3
    layer2 = ConvolutionalLayer(activation, filter_size, num_filters,
                                pooling_size=(pooling_size, pooling_size),
                                pooling_step=pooling_step,
                                pad = pad,
                                name="layer2")

    conv_layers = [layer0, layer1, layer2]
    convnet = ConvolutionalSequence(conv_layers, num_channels= 3,
                                    image_size=(32, 32),
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Uniform(width=0.1)
                                    )
    convnet.initialize()
    output_dim = np.prod(convnet.get_dim('output'))
    output_conv = Flattener().apply(convnet.apply(x))

    ######### SUITE ##################

    nbr_classes_to_predict = 10 # because it's SVHN

    assert L_nbr_hidden_units[0] == output_dim
    #nbr_hidden_units = [output_dim, 1024, 512, 512]
    padded_drop_mlp = [0.0] + drop_mlp

    L_nbr_hidden_units_left = [ n-int(p*n) for (n,p) in zip(L_nbr_hidden_units, padded_drop_mlp) ]
    # add afterwards the final number of hidden units to the number of classes to predict
    mlp_dim_pairs = zip(L_nbr_hidden_units_left, L_nbr_hidden_units_left[1:]) + [(L_nbr_hidden_units_left[-1], nbr_classes_to_predict)]

    print "mlp_dim_pairs"
    print mlp_dim_pairs

    # MLP
    sequences_mlp = []
    mlp_layer = []

    num_pieces = 5
    mlp_layer0 = LinearMaxout(mlp_dim_pairs[0][0], mlp_dim_pairs[0][1],
                        num_pieces=num_pieces,
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer4")
    mlp_layer0.initialize()

    mlp_layer1 = Linear(mlp_dim_pairs[1][0], mlp_dim_pairs[1][1],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer5")
    mlp_layer1.initialize()

    output_mlp_0 = mlp_layer0.apply(output_conv)
    output_mlp_1 = Softmax().apply(mlp_layer1.apply(output_mlp_0))
    output_full = output_mlp_1

    cost = CategoricalCrossEntropy().apply( y.flatten(), output_full )
    cost.name = 'cost'
    cg = ComputationGraph(cost)

    diagnostic_output = output_full[0,:]
    diagnostic_output.name = "diagnostic_output"


    # put names
    params, names = build_params(x, T.matrix(), conv_layers, mlp_layer)
    # test computation graph
    error_rate_brick = MisclassificationRate()
    error_rate = error_rate_brick.apply(y.flatten(), output_full)

    #error_rate = errors(output_full, y)
    error_rate.name = 'error'
    ###step_rule = Momentum_dict(learning_rate, momentum, params=params)
    
    step_rule = Momentum(step_flavor['learning_rate'], step_flavor['momentum'])

    # DEBUG sqns le serveur
    """
    dict_params = step_rule.velocities
    for param_m in dict_params :
        print param_m
        names.append(param_m)
    for param in params:
        dict_params[param.name] = param
    """
    dict_params = None
    return (cg, error_rate, cost, step_rule, names, dict_params, diagnostic_output)

def build_training(cg, error_rate, cost, step_rule,
                   dataset_hdf5_file=None,
                    batch_size=256, dropout_bis=None, nb_epochs=1, saving_path=None, nb_iteration=0,
                   server_update_extension=None, checkpoint_interval_nbr_batches=10, diagnostic_output=None):

    # ici ou avant
    if dropout_bis is not None:
        # apply a second level of dropout
        inputs = VariableFilter(roles=[INPUT])(cg.variables)
        cg_dropout = cg
        for input_ in inputs:
            if input_ in dropout_bis:
                cg_dropout = apply_dropout(cg_dropout, [input_], dropout_bis[input_.name]) 
        # we can do several dropout
        cg = cg_dropout
    
    train_set = H5PYDataset(dataset_hdf5_file, which_set='train')


    data_stream = DataStream.default_stream(
             #train_set, iteration_scheme=SequentialScheme(100*batch_size, batch_size))
             train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
    
    # monitoring the train set
    data_stream_monitor = DataStream.default_stream(
             train_set, iteration_scheme=SequentialScheme(10*batch_size, batch_size))
             #train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
     

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    
    valid_set = H5PYDataset(dataset_hdf5_file, which_set='test')

    data_stream_valid =DataStream.default_stream(
            #valid_set, iteration_scheme=SequentialScheme(10000, batch_size))
            valid_set, iteration_scheme=SequentialScheme(5000, batch_size))
            #valid_set, iteration_scheme=SequentialScheme(valid_set.num_examples, batch_size))
    
    """
    test_set = H5PYDataset(database['test'], which_set='test')
    data_stream_test =DataStream.default_stream(
             #test_set, iteration_scheme=SequentialScheme(100*batch_size, batch_size))
             test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size))
    
    monitor_test = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="test", every_n_batches=checkpoint_interval_nbr_batches)
    """
    monitor_valid = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_valid, prefix="test", every_n_batches=checkpoint_interval_nbr_batches)

    monitor_train = DataStreamMonitoring(
        variables=[cost, diagnostic_output], data_stream=data_stream_monitor, prefix="train", every_n_batches=checkpoint_interval_nbr_batches)

    extensions = [monitor_valid, monitor_train, 
                            FinishAfter(after_n_epochs=1000), Printing(every_n_batches=checkpoint_interval_nbr_batches),
                  ]
    if server_update_extension is not None:
        extensions.append(server_update_extension)
    if saving_path is not None:
        extensions += [Dump(saving_path+"/"+str(nb_iteration))]

    main_loop = MainLoop(data_stream=data_stream,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)
    main_loop.run()

if __name__ == '__main__':
    drop=0.5*np.zeros((16,))
    learning_rate = 1e-4
    momentum=0.99
    drop_conv = [0.0, 0.0, 0.0, 0.0]
    drop_mlp = [0.0, 0.0, 0.0]    
    batch_size = 32
    
    dataset_hdf5_file = "/mnt/raid5vault6/tmp/ML/SVHN/ninjite_h5/easy_svhn.h5"

    #path="./"
    cg, error_rate, cost, step_rule, names, _ = build_architecture( drop_conv=drop_conv,
                                                                    drop_mlp=drop_mlp,
                                                                    learning_rate=learning_rate,
                                                                    momentum=momentum,
                                                                    dataset_hdf5_file=dataset_hdf5_file)
    
    #build_training(cg, error_rate, cost, step_rule,
    #                batch_size=batch_size, path=path)
    checkpoint_interval_nbr_batches = 10
    build_training(cg, error_rate, cost, step_rule,
        batch_size=batch_size, dropout_bis = None,
        server_update_extension=None,
        dataset_hdf5_file=dataset_hdf5_file,
        checkpoint_interval_nbr_batches=checkpoint_interval_nbr_batches)
