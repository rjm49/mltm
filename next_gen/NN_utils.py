from keras import backend as K
from keras.constraints import Constraint
from keras.engine.topology import Layer
from keras import initializers

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, min_w=0, max_w=4):
        self.min_w = min_w
        self.max_w = max_w

    def __call__(self, p):
        return K.clip(p, self.min_w, self.max_w)

    def get_config(self):
        return {
#                 'name': self.__class__.__name__,
                'min_w': self.min_w,
                'max_w': self.max_w }
      
from keras import constraints
from keras import regularizers
import numbers
import math
# class BigTable(Layer):

#     def __init__(self, _dim, min_w, max_w, regulariser=None, **kwargs):
#         self.dim = _dim
#         self.hilo = kwargs["init_hilo"]
#         kwargs.pop('init_hilo', None)
#         self.limits = (min_w, max_w)
#         kc =kernel_constraint=WeightClip(min_w, max_w)
#         self.kernel_constraint= constraints.get(kc)
#         self.reg = regulariser
#         super(BigTable, self).__init__(**kwargs)
   
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         min_w, max_w = self.limits
# #         initialiser = initializers.RandomUniform(av_w*0.99, av_w*1.01)  #.RandomUniform(min_w, max_w)
#         if self.hilo is not None:
#             if type(self.hilo) is tuple:
#                 initialiser = initializers.RandomUniform(self.hilo[0], self.hilo[1])
        
#             elif self.hilo=="hi":
#                 initialiser = initializers.RandomUniform(max_w*.9, max_w)
# #             elif self.hilo=="lo":
# #                 initialiser = initializers.RandomUniform(min_w, min_w*1.1)
# #             elif self.hilo=="av":
# #                 av_w = (min_w + max_w)/2.0
# #                 initialiser = initializers.RandomUniform(av_w*0.9, av_w*1.1)
# #             elif self.hilo=="all":
# #                 initialiser = initializers.RandomUniform(min_w, max_w)
#             else:
#                 w = self.hilo
# #                 if w == 0:
# #                     initialiser = initializers.RandomUniform(w, w+0.5)
# #                     print(initialiser)
# #                 else:
#                 initialiser = initializers.RandomUniform(w-.1, w+.1)
#         else:
#             if min_w == -math.inf or max_w == math.inf:
#                 print("infs -> glorot init")
#                 initialiser = initializers.glorot_uniform()
#             else:
#                 initialiser = initializers.RandomUniform(min_w, max_w)
#                 print("Blank init'd:",initialiser)
            
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=(self.dim),
#                                       initializer= initialiser,
#                                       regularizer=self.reg,
#                                       trainable=True,
#                                       constraint=self.kernel_constraint,
#                                      )
#         print("kk", self.kernel.shape)
#         super(BigTable, self).build(input_shape)  # Be sure to call this at the end

#     def call(self, selector):
#         print("selector shape", selector.shape)
#         selector = K.flatten(selector)
#         print("flat selector shape", selector.shape)
#         print("call kk", self.kernel.shape)
# #         selector = tf.Print(selector, [selector], message="selector is:", first_n=-1, summarize=1024)
#         rows = K.gather(self.kernel, selector)
# #         rows = tf.Print(rows, [rows], message="row is:", first_n=-1, summarize=1024)
#         print("'rows' shape,",rows.shape)
#         return rows

#     def compute_output_shape(self, input_shape):
#         return ((None, self.dim[1]))
