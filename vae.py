# code from https://github.com/udibr/VAE/blob/master/VAE.py

#!/usr/bin/env python
import logging
from argparse import ArgumentParser

import theano
from theano import tensor
import theano.tensor as T

import blocks
from blocks.bricks import MLP, Tanh, Rectifier
from blocks.initialization import Constant, NdarrayInitialization, Sparse, Orthogonal
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop

import fuel
import os
from fuel.datasets.hdf5 import H5PYDataset

floatX = theano.config.floatX
import numpy as np
import cPickle as pickle

#-----------------------------------------------------------------------------
from blocks.bricks import Initializable, Random, Linear
from blocks.bricks.base import application

class Qlinear(Initializable):
    """
    brick to handle the intermediate layer of an Autoencoder.
    In this brick a simple linear mix is performed (a kind of PCA.)
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qlinear, self).__init__(**kwargs)

        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_transform]

    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError

    @application(inputs=['x'], outputs=['z', 'kl_term'])
    def sample(self, x):
        """Sampling is trivial in this case
        """
        mean = self.mean_transform.apply(x)

        z = mean

        # Calculate KL
        batch_size = x.shape[0]
        kl = T.zeros((batch_size,),dtype=floatX)

        return z, kl

    @application(inputs=['x'], outputs=['z'])
    def mean_z(self, x):
        return self.mean_transform.apply(x)


class Qsampler(Qlinear, Random):
    """
    brick to handle the intermediate layer of an Autoencoder.
    The intermidate layer predict the mean and std of each dimension
    of the intermediate layer and then sample from a normal distribution.
    """
    # Special brick to handle Variatonal Autoencoder statistical sampling
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qsampler, self).__init__(input_dim, output_dim, **kwargs)

        self.prior_mean = 0.
        self.prior_log_sigma = 0.

        self.log_sigma_transform = Linear(
                name=self.name+'_log_sigma',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children.append(self.log_sigma_transform)

    @application(inputs=['x'], outputs=['z', 'kl_term'])
    def sample(self, x):
        """Return a samples and the corresponding KL term

        Parameters
        ----------
        x :

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x)
        kl : tensor.vector
            KL(Q(z|x) || P_z)

        """
        mean = self.mean_transform.apply(x)
        log_sigma = self.log_sigma_transform.apply(x)

        batch_size = x.shape[0]
        dim_z = self.get_dim('output')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
                    size=(batch_size, dim_z),
                    avg=0., std=1.)
        z = mean + tensor.exp(log_sigma) * u

        # Calculate KL
        kl = (
            self.prior_log_sigma - log_sigma
            + 0.5 * (
                tensor.exp(2 * log_sigma) + (mean - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_log_sigma)
            - 0.5
        ).sum(axis=-1)

        return z, kl
#-----------------------------------------------------------------------------


class VAEModel(Initializable):
    """
    A brick to perform the entire auto-encoding process
    """
    def __init__(self,
                    encoder_mlp, sampler,
                    decoder_mlp, **kwargs):
        super(VAEModel, self).__init__(**kwargs)

        self.encoder_mlp = encoder_mlp
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp

        self.children = [self.encoder_mlp, self.sampler, self.decoder_mlp]

    def get_dim(self, name):
        if name in ['z', 'z_mean', 'z_log_sigma']:
            return self.sampler.get_dim('output')
        elif name == 'kl':
            return 0
        else:
            super(VAEModel, self).get_dim(name)

    @application(inputs=['features'], outputs=['reconstruction', 'kl_term'])
    def reconstruct(self, features):
        enc = self.encoder_mlp.apply(features)
        z, kl = self.sampler.sample(enc)

        x_recons = self.decoder_mlp.apply(z)
        x_recons.name = "reconstruction"

        kl.name = "kl"

        return x_recons, kl

    @application(inputs=['features'], outputs=['z', 'enc'])
    def mean_z(self, features):
        enc = self.encoder_mlp.apply(features)
        z = self.sampler.mean_z(enc)

        return z, enc

    @application(inputs=[], outputs=['x'])
    def generate_sample(self):
        # generate z given the prior
        # pay attention to the seed
        batch_size = 1
        dim_z = self.sampler.get_dim('output')
        z = self.sampler.theano_rng.normal(
                    size=(batch_size, dim_z),
                    avg=self.sampler.prior_mean, std= T.exp(self.sampler.prior_log_sigma))

        return self.decoder_mlp.apply(z)

#-----------------------------------------------------------------------------

def shnum(value):
    """ Convert a float into a short tag-usable string representation. E.g.:
        0 ->
        0.1   -> 11
        0.01  -> 12
        0.001 -> 13
        0.005 -> 53
    """
    if value <= 0.:
        return '0'
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, -exp)

