#! /bin/python
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text, ctc_to_char
from IPython.display import Audio
from sample_models import *
from beamsearch import ctcBeamSearch
from tokenpassing import ctcTokenPassing
from char_map import index_map
from languagemodel import LanguageModel

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

    # loading language model
    alphabet = ''.join(index_map.values())
    language_model = LanguageModel('data/word/corpus.txt', alphabet)

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
        elif partition == 'test':
            transcr = 'hello how are you'
            audio_path = '../datasets/AgentExpress/hello.wav.wav'
            data_point = data_gen.normalize(data_gen.featurize(audio_path))
            #print(data_point)
        else:
            raise Exception('Invalid partition!  Must be "train" or "validation"')
            
        
        prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    
        output_length = [input_to_softmax.output_length(data_point.shape[0])]
        
        #print(K.eval(K.ctc_decode(prediction, output_length, False, 30, 1)[0][0]))
        pred_ints = (K.eval(K.ctc_decode(
                    prediction, output_length)[0][0])+1).flatten().tolist()
        
        # play the audio file, and display the true and predicted transcriptions
        print('-'*80)
        Audio(audio_path)
        print('ground_truth:' + ' '*4 + transcr)
        print('best_path:'+ ' '*7 +''.join(int_sequence_to_text(pred_ints)))
        pred_beam = ctcBeamSearch(prediction[0], alphabet, None)
        print('beam_search:' + ' '*5  + pred_beam)
        pred_beam_lm = ctcBeamSearch(prediction[0], alphabet, language_model)
        print('beam_search_lm:' + ' '*2  + pred_beam_lm)
        pred_token = ctcTokenPassing(prediction[0], alphabet, language_model.getWordList())
        print('token_passing:' + ' '*3  + pred_token)
        


if __name__ == '__main__':
    get_predictions(indexes=[0, 5, 6, 7, 8, 9], 
                partition='validation',
                input_to_softmax=final_model(input_dim=13, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11,
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200), 
                model_path='results/model_final_500.h5')
