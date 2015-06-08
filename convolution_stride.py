"""
Convolutional Layer zith stride for the pooling

"""
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans
from blocks.bricks.conv import ConvolutionalActivation, MaxPooling
from blocks.bricks import Rectifier, Activation, Linear



class Padding(Activation):
    
    @lazy(allocation=['pad'])
    def __init__(self, pad=0, **kwargs):
        super(Padding, self).__init__(**kwargs)
        self.pad = pad

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        apply padding on the input
        input is a 4D tensor
        """
        if self.pad :
            shape_ = input_.shape
            shape = (shape_[0], shape_[1], shape_[2] + 2*self.pad, shape_[3]+2*self.pad)
            output = T.zeros(shape)
            output = T.set_subtensor(output[:,:,self.pad:shape[2]-self.pad,self.pad:shape[3]-self.pad],
                                    input_)
        else:
            output=input_

        return output


class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.
    .. todo::
       Mean pooling.
    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling. Needed for ``__init__``.
    See Also
    --------
    :class:`Convolutional` : Documentation of convolution arguments.
    :class:`MaxPooling` : Documentation of pooling arguments.
    Notes
    -----
    Uses max pooling.
    """
    @lazy(allocation=['filter_size', 'num_filters', 'pooling_size',
                      'num_channels'])
    def __init__(self, activation, filter_size, num_filters, pooling_size,
                 num_channels, conv_step=(1, 1), pooling_step=None,
                 batch_size=None, image_size=None, border_mode='valid',
                 tied_biases=False, pad=0, **kwargs):
        self.convolution = ConvolutionalActivation(activation)
        self.pooling = MaxPooling()
        self.padding = Padding(pad)

        super(ConvolutionalLayer, self).__init__(
            application_methods=[self.padding.apply,
                                 self.convolution.apply,
                                 self.pooling.apply], **kwargs)
        self.convolution.name = self.name + '_convolution'
        self.pooling.name = self.name + '_pooling'

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.pooling_size = pooling_size
        self.conv_step = conv_step
        self.pooling_step = pooling_step
        self.batch_size = batch_size
        self.border_mode = border_mode
        # change image_size given the pad
        self.pad = pad
        self.image_size = image_size
        self.tied_biases = tied_biases
        self.activation = activation

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        apply padding on the input
        input is a 4D tensor
        """
        return self.pooling.apply(self.convolution.apply(self.padding.apply(input_)))

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'num_channels',
                     'batch_size', 'border_mode',
                     'tied_biases']:
            setattr(self.convolution, attr, getattr(self, attr))
        for attr in ['image_size']:
            image_size = getattr(self, attr)
            image_size = (image_size[0] + 2*self.pad, image_size[1] + 2*self.pad)
            setattr(self.convolution, attr, image_size)

        self.convolution.step = self.conv_step
        self.convolution._push_allocation_config()
        if self.image_size is not None:
            pooling_input_dim = ((self.convolution.num_filters/2,) +
                    ConvOp.getOutputShape(self.convolution.image_size, self.convolution.filter_size,
                                          self.convolution.step, self.convolution.border_mode))

        else:
            pooling_input_dim = None
        self.pooling.input_dim = pooling_input_dim
        self.pooling.pooling_size = self.pooling_size
        self.pooling.step = self.pooling_step
        self.pooling.batch_size = self.batch_size
    
    
    def get_dim(self, name):
        if name == 'input_':
            temp = self.convolution.get_dim('input_')
            return (temp[0] -2*self.pad, temp[1] - 2*self.pad)

        if name == 'output':
            pooling_output = self.pooling.get_dim('output')     
            return pooling_output


class Maxout_(Brick):
    """Maxout pooling transformation.
    A brick that does max pooling over groups of input units. If you use
    this code in a research project, please cite [GWFM13]_.
    .. [GWFM13] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
       Courville, and Yoshua Bengio, *Maxout networks*, ICML (2013), pp.
       1319-1327.
    Parameters
    ----------
    num_pieces : int
        The size of the groups the maximum is taken over.
    Notes
    -----
    Maxout applies a set of linear transformations to a vector and selects
    for each output dimension the result with the highest value.
    """
    @lazy(allocation=['num_pieces'])
    def __init__(self, num_pieces, **kwargs):
        super(Maxout_, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the maxout transformation.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input
        """
        input_ = input_.dimshuffle((0, 2, 3, 1))
        last_dim = input_.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)] +
                     [output_dim, self.num_pieces])
        output = T.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output.dimshuffle((0, 3, 1, 2))
