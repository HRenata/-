from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, LSTM,
    TimeDistributed, Activation, Bidirectional, Dropout)

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def final_model(filters, kernel_size, conv_stride,
    conv_border_mode, units, input_dim=13, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add batch normalization
    # Add a recurrent layer
    bidir_rnn = Bidirectional(LSTM(units*3, return_sequences=True,
                    name='bidir_rnn_1'), merge_mode='concat')(bn_cnn)
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(bidir_rnn)

    
    bidir_rnn_2 = Bidirectional(LSTM(units*2,
        return_sequences=True, name='bidir_rnn_2'), merge_mode='concat')(bn_rnn_1)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(bidir_rnn_2)
    drop_2 = Dropout(0.1)(bn_rnn_2)

    bidir_rnn_3 = Bidirectional(LSTM(units*1,
        return_sequences=True, name='bidir_rnn_3'), merge_mode='concat')(drop_2)
    bn_rnn_3 = BatchNormalization(name='bn_rnn_3')(bidir_rnn_3)
    drop_3 = Dropout(0.3)(bn_rnn_3)

    bidir_rnn_4 = Bidirectional(LSTM(units*1,
        return_sequences=True, name='bidir_rnn_4'), merge_mode='concat')(drop_3)
    bn_rnn_4 = BatchNormalization(name='bn_rnn_4')(bidir_rnn_4)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_4)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model