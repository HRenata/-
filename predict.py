#! /bin/python
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text, ctc_to_char
from IPython.display import Audio
from sample_models import *

def get_predictions(indexes, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=False, mfcc_dim=13)
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)

    for index in indexes:
        # obtain the true transcription and the audio features 
        if partition == 'validation':
            transcr = data_gen.valid_texts[index]
            audio_path = data_gen.valid_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(audio_path))
        elif partition == 'train':
            transcr = data_gen.train_texts[index]
            audio_path = data_gen.train_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(audio_path))
        else:
            raise Exception('Invalid partition!  Must be "train" or "validation"')
            
        
        prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    
        output_length = [input_to_softmax.output_length(data_point.shape[0])]
        #print(ctc_to_char(prediction))
        print(K.eval(K.ctc_decode(prediction, output_length, False, 30, 1)[0][0]))
        pred_ints = (K.eval(K.ctc_decode(
                    prediction, output_length)[0][0])+1).flatten().tolist()
        
        # play the audio file, and display the true and predicted transcriptions
        print('-'*80)
        Audio(audio_path)
        print('True transcription:\n' + '\n' + transcr)
        print('-'*80)
        print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
        print('-'*80)


if __name__ == '__main__':
    get_predictions(indexes=[0,1,2,3,4], 
                partition='validation',
                input_to_softmax=final_model(input_dim=13, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11,
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200), 
                model_path='results/model_final_500.h5')
    