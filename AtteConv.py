from keras.layers import Flatten, Dropout, Dense, Lambda, ZeroPadding2D, MaxPooling2D, Concatenate, Conv2D, LSTM, Bidirectional, Activation
from keras import backend as K

from cnnatte import Attention


def build_cnn(input, dropout, kernel_sizes, num_stages, num_filters, pool_sizes, attention=False, context='same',
              fully_connected_dimension=1000):
    '''
    This method builds a cnn with the given parameters
    :param input: Output of preceding layer
    :param dropout: dropout-rate
    :param kernel_sizes: list of lists, inner list defines different kernel-size per stage, outer list defines different stages
    :param num_stages: describes the number of stages
    :param num_filters: list, different number of filters can be used per stage
    :param pool_sizes: list, different pool-sizes can be used per stage
    :param attention: defines if attention should be applied
    :param context: defines the context. 'same' means self-attention. Otherwise a layer-output can be given to apply query-attention
    :return: input & output of the encoder
    '''
    input_cpy = input
    for i in range(num_stages):
        if i > 0:
            attention = False
        input_cpy = _build_stage(kernel_sizes[i], input_cpy, num_filters[i], context, pool_sizes[i], attention)

    flatten = Flatten()(input_cpy)
    dropout = Dropout(dropout)(flatten)
    fully_connected = Dense(units=fully_connected_dimension)(dropout)

    return input, fully_connected


def _build_stage(kernel_sizes, pre_layer, num_filters, context, pool_size, attention):
    convs = []
    for size in kernel_sizes:
        reshape = Lambda(lambda x: K.expand_dims(x, 3))(pre_layer)
        if size % 2 == 0:
            padded_input = ZeroPadding2D(padding=((int(size / 2), int(size / 2) - 1), (0,0)))(reshape)
        else:
            padded_input = ZeroPadding2D(padding=((int(size / 2), int(size / 2)), (0,0)))(reshape)

        conv = Conv2D(num_filters, (size, int(reshape.shape[2])), activation='relu', padding='valid')(padded_input)
        convs.append(conv)

    if len(convs) > 1:
        all_filters = Concatenate(axis=3)(convs)
    else:
        all_filters = convs[0]

    if attention:
        all_filters = Lambda(lambda x: K.squeeze(x, 2))(all_filters)
        if context == 'same':
            attentive_context = Attention(before=True)([all_filters, all_filters])
        else:
            attentive_context = Attention(before=True)([all_filters, context])
        attentive_context = Lambda(lambda x: K.expand_dims(x, axis=2))(attentive_context)
        reshape = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(attentive_context)
    else:
        reshape = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(all_filters)

    if pool_size > int(reshape.shape[1]):
        pool_size = int(reshape.shape[1])
    filtered = MaxPooling2D(pool_size=(pool_size, 1))(reshape)

    reshape = Lambda(lambda x: K.squeeze(x, 3))(filtered)
    return reshape
