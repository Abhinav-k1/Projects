import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def get_path(track_id):
    """
    Returns the path to the mp3.
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join('../AI/fma_small', tid_str[:3], tid_str + '.mp3')


def get_track_IDs():
    """
    Get track IDs from the mp3s in a directory.
    """
    ids = []
    for _, dirnames, files in os.walk('../AI/fma_small'):
        if dirnames == []:
            ids.extend(int(file[:-4]) for file in files)
    return ids

def get_track_data():
    """
    Reads information from tracks.csv file
    """
    tracks = pd.read_csv('../AI/fma_small/tracks.csv', index_col=0, header=[0, 1])
    keep_cols = [('set', 'split'),
    ('set', 'subset'),('track', 'genre_top')]
    df = tracks[keep_cols]
    df = df[df[('set', 'subset')] == 'small']
    df['track_id'] = df.index
    df.columns = ['split', 'x', 'genre', 'track_id']
    df = df.drop(columns=['x'])
    return df

def create_spectogram(track_id):
    """
    Creates a melspectogram.
    """
    filename = get_path(track_id)
    y, sr = librosa.load(filename)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 1024)
    spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    return spectrogram[:, 215:343]
    
def create_spectogram1(track_id):
    """
    Creates a melspectogram of a different slice of the track.
    """
    filename = get_path(track_id)
    y, sr = librosa.load(filename)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 1024)
    spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    return spectrogram[:, 473:601]

def plot_spectogram(track_id, genre):
    """
    Plots a melspectogram.
    """
    spectrogram = create_spectogram(track_id)
    print(spectrogram.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency (Hz)")
    plt.title(genre)
    plt.show()
    
def resize_data(image):
    """
    Use bi-linear interpolation to skrink the image.
    """
    image = tf.reshape(image, shape = [128,128,1])
    image = tf.image.resize(image, size = [64, 64], preserve_aspect_ratio=False,
    antialias=False, name=None)
    image = tf.reshape(image, shape = [64,64])
    proto_tensor = tf.make_tensor_proto(image)
    return tf.make_ndarray(proto_tensor)

genre_dict = {'Electronic':0,  'Hip-Hop':1, 
               'Instrumental':2, 'Rock': 3, 'Experimental':4, 'Folk':5, 'International':6, 'Pop' :7}

def convert_data(df):
    """
    Converts each track in a dataframe from sound to a spectogram.
    """
    y = []
    X = np.empty((0, 64, 64))
    count = 0
    
    for index, row in df.iterrows():
        try:
            count += 1
            spect = create_spectogram(int(row['track_id']))
            spect = resize_data(spect)
            X = np.append(X, [spect], axis=0)
            y.append(genre_dict[row['genre']])
            if count % 100 == 0:
                print("Processed: ", count)
        except:
            continue
    
    Y = np.array(y)
    print(X.shape, Y.shape)
    return X, Y

def standardise_data(X, Y):
    """
    Standardises X and one_hot encodes Y.
    """
    standard = tf.image.per_image_standardization(X)
    X_ = tf.convert_to_tensor(standard)
    Y_ = tf.convert_to_tensor(Y)
    Y_ = tf.one_hot(Y_, depth = 8)
    return (X_, Y_)

def get_test_data():
    """
    Loads test data.
    """
    npzfile = np.load('data/test_dat.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    X_t1, Y_t1 = standardise_data(X, Y)
    npzfile1 = np.load('data/test_dat1.npz')
    X1 = npzfile1['arr_0']
    Y1 = npzfile1['arr_1']
    X_t2, Y_t2 = standardise_data(X, Y)
    X_test = tf.concat([X_t1, X_t2], 0)
    Y_test = tf.concat([Y_t1, Y_t2], 0)
    return (X_test, Y_test)

def get_valid_data():
    """
    Loads validation data.
    """
    npzfile = np.load('data/validation_dat.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    X_t1, Y_t1 = standardise_data(X, Y)
    npzfile1 = np.load('data/validation_dat1.npz')
    X1 = npzfile1['arr_0']
    Y1 = npzfile1['arr_1']
    X_t2, Y_t2 = standardise_data(X, Y)
    X_valid = tf.concat([X_t1, X_t2], 0)
    Y_valid = tf.concat([Y_t1, Y_t2], 0)
    return (X_valid, Y_valid)