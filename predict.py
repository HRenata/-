import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_predictions(audio_path, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        audio_path (str): path to audio file
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator(mfcc_dim=13)
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)

    data_point = data_gen.normalize(data_gen.featurize(audio_path))

    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))

    output_length = [input_to_softmax.output_length(data_point.shape[0])]

    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()

    print('PREDICTION:')
    print(int_sequence_to_text(pred_ints))
