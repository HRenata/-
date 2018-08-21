#! /bin/python

from data_generator import vis_train_features
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sample_models import *
from train_utils import train_model
import sys


def load_data():
    vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()
    return vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path 

def load_env():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

def model_0():
    model = simple_rnn_model(input_dim=161)
    train_model(input_to_softmax=model,
            pickle_path='model_0.pickle',
            save_model_path='model_0.h5',
            spectrogram=True)
    pass

def model_1():
    model_1 = rnn_model(input_dim=13, # change to 13 if you would like to use MFCC features
                    units=200,
                    activation='relu')
    train_model(input_to_softmax=model_1,
            pickle_path='model_1.pickle',
            save_model_path='model_1.h5',
            spectrogram=False)
    pass

def model_2():
    model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11,
                        conv_stride=1,
                        conv_border_mode='valid',
                        units=200)
    train_model(input_to_softmax=model_2,
            pickle_path='model_2.pickle',
            save_model_path='model_2.h5',
            spectrogram=True)
    pass

def model_3():
    model_3 = deep_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                         units=200,
                         recur_layers=3)
    train_model(input_to_softmax=model_3,
            pickle_path='model_3.pickle',
            save_model_path='model_3.h5',
            spectrogram=True)
    pass

def model_4():
    model_4 = bidirectional_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                                  units=200)
    train_model(input_to_softmax=model_4,
            pickle_path='model_4.pickle',
            save_model_path='model_4.h5',
            sort_by_duration=True,
            spectrogram=True)
    pass
def model_4_b():
    model_4 = bidirectional_rnn_model(input_dim=13, # change to 13 if you would like to use MFCC features
                                  units=200)
    train_model(input_to_softmax=model_4,
            pickle_path='model_4_b.pickle',
            save_model_path='model_4_b.h5',
            sort_by_duration=True,
            spectrogram=False)

def model_final():
    model_final = final_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=400,
                        kernel_size=11,
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=400)
    train_model(input_to_softmax=model_final,
            pickle_path='model_final.pickle',
            save_model_path='model_final.h5',
            spectrogram=True)
    



if __name__=='__main__':
    print(sys.argv)

    load_env()

    command = sys.argv[1]

    if command == 'model0':
        model_0()
    if command == 'model1':
        model_1()
    if command == 'model2':
        model_2()
    if command == 'model3':
        model_3()
    if command == 'model4':
        model_4()
    if command == 'model4b':
        model_4_b()
    if command == 'final':
        model_final()
    
    
    
    
    #load_data()
    #load_env()
    #model_4()


