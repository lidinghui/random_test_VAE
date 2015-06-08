import blocks
from blocks.extensions import SimpleExtension
import theano.tensor as T


class Clip_param(SimpleExtension):

    def __init__(self, params, **kwargs):
        super(Clip_param, self).__init__(every_n_batches=1, **kwargs)
        self.params = []
        for p in params:
            if p.name in ['layer_0_W', 'layer_1_W', 'layer_2_W', "layer_3_W", "layer_4_W"]:
                self.params.append(p)

    def do(self, which_callback, *args):

        for p in self.params:
            if p.name in ['layer_0_W', 'layer_1_W', 'layer_2_W']:
                norm_kernel = (p.norm(2, axis=(1,2,3)))
                if p.name =="layer_0_W":
                    constant = 0.9
                else:
                    constant = 1.9365

                normalisation = T.clip(norm_kernel, 0., constant)
                norm_kernel = T.clip(norm_kernel, 1e-10, T.max(norm_kernel))
                normalisation = normalisation.dimshuffle((0, 'x', 'x', 'x'))
                norm_kernel = normalisation.dimshuffle((0, 'x', 'x', 'x'))
                p = (p/norm_kernel)*normalisation
            """
            if p.name in ["layer_3_W", "layer_4_W"]:
                norm_col = p.norm(2, axis=1)
                if p.name =="layer_3_W":
                    constant = 1.9
                if p.name =="layer_4_W":
                    constant = 1.9365
                normalisation = T.clip(norm_col, 0., constant)
                norm_col = T.clip(norm_col, 1e-10, T.max(norm_col))
                normalisation = normalisation.dimshuffle(('x', 0))
                norm_col = norm_col.dimshuffle(('x', 0))
                p = (p/norm_col)*normalisation
            """
