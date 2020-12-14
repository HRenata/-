import librosa
from data_generator import AudioGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from glob import glob
import _pickle as pickle

def plot_raw_audio(vis_audio_path):
    "plot the raw audio signal"
    vis_raw_audio, _ = librosa.load(vis_audio_path)
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    steps = len(vis_raw_audio)
    ax.plot(np.linspace(1, steps, steps), vis_raw_audio)
    plt.title('Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def plot_mfcc_feature(audiopath):
    "plot the mel-spectrogram for audio signal"
    # load the train and test data
    data_gen = AudioGenerator(mfcc_dim=13)
    data_gen.load_train_data()
    data_gen.load_validation_data()

    vis_mfcc_feature = data_gen.normalize(data_gen.featurize(audiopath))
    # plot the normalized spectrogram
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(0, 13, 2), minor=False)
    plt.show()


def plot_train_valid_loss():
    "plot the training and validation loss"
    # obtain the paths for the saved model history
    all_pickles = glob("results/model_final_2020.pickle")

    # extract the loss history for each model
    valid_loss = pickle.load(open(all_pickles[0], "rb"))['val_loss']
    train_loss = pickle.load(open(all_pickles[0], "rb"))['loss']

    # save the number of epochs used to train each model
    num_epochs = len(valid_loss)

    fig = plt.figure(figsize=(16, 5))

    # plot the training loss vs. epoch for each model
    ax = fig.add_subplot(121)
    ax.plot(np.linspace(1, num_epochs, num_epochs),
            train_loss, label="training loss")
    ax.plot(np.linspace(1, num_epochs, num_epochs),
            valid_loss, label="validation loss")

    # clean up the plot
    ax.legend()
    ax.set_xlim([1, num_epochs])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()