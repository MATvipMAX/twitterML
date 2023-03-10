from keras.layers import Lambda, Reshape, RepeatVector, Concatenate, Conv1D, Activation
from keras.layers import Layer
from keras import activations
from keras import backend as K
import tensorflow as tf


class Attention(Layer):

    def __init__(self, kernel_activation='hard_sigmoid', before=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.kernel_activation = activations.get(kernel_activation)
        K.set_floatx('float32')
        self.before = before

    def build(self, input_shape):
        self.num_words = input_shape[0][1]
        #self.em_dim = input_shape[0][2]

        super(Attention, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        if self.before:
            return input_shape
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.before:
            return input_shape
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        text = x[0]
        context = x[1]

        length = 1
        for i in range(len(context.shape)):
            if i > 0:
                length *= int(context.shape[i])
        context = Lambda(lambda x: Reshape((length,))(x))(context)

        context_repeated = RepeatVector(self.num_words)(context)
        merged = Concatenate(axis=2)([context_repeated, text])
        scores = Conv1D(1,1)(merged)
        weights = Activation(activation='softmax')(scores)
        #weighted = K.transpose(tf.multiply(K.transpose(text), weights))

        if not self.before:
            weigthed = K.batch_dot(K.permute_dimensions(text, (0,2,1)), weights)
            return K.squeeze(weigthed, 2)


        weigthed = tf.multiply(text, weights)
        return weigthed

    def get_config(self):
        config = {'kernel_activation': activations.serialize(self.kernel_activation),
                  'before': self.before}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
