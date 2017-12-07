from dataset import *
from music_tagger_cnn import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.engine import Model, Input
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
K.set_image_dim_ordering('th')
from scipy import misc
from keras.layers.convolutional import ZeroPadding2D
from skimage.transform import resize
from load_dataset import *
from music_tagger_cnn import *
from sklearn.preprocessing import LabelEncoder
from load_dataset import *
from dataset import *
from keras.callbacks import Callback
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn.metrics import confusion_matrix
import json
import matplotlib
matplotlib.use('Agg') 


def plot_metrics(history):

    print(history.history.keys())

    fig = matplotlib.pyplot.figure(1)

    # summarize history for accuracy

    matplotlib.pyplot.subplot(211)
    matplotlib.pyplot.plot(history.history['acc'])
    matplotlib.pyplot.plot(history.history['val_acc'])
    matplotlib.pyplot.title('model accuracy')
    matplotlib.pyplot.ylabel('accuracy')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    matplotlib.pyplot.subplot(212)
    matplotlib.pyplot.plot(history.history['loss'])
    matplotlib.pyplot.plot(history.history['val_loss'])
    matplotlib.pyplot.title('model loss')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')
    fig.savefig('metrics_RNN.png', dpi=fig.dpi)

# Load dataset

data = dataset('/imatge/epresas/music_genre/data/genres', 10, 100)
data.create()
print(data.X_train.shape)

# Build the RNN model

input_tensor = Input(shape=(1, 159, 13))

model = Sequential()
model.add(LSTM(12, input_shape= (1,44100)))

model.add(Dense(10, activation='softmax'))
model.summary()
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
'''


sgd = SGD(lr=0.1, momentum=0, decay=0.002, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(data.X_train, data.labels_train,
                              validation_data=(data.X_val, data.labels_val), nb_epoch=100,
                              batch_size=5,verbose=1)
#loss_and_metrics = model.evaluate(data.X_val, data.labels_val, batch_size=5)
plot_metrics(history)
predict_labels=model.predict(data.X_val)
predictions = []
orig_predictions=[]

for prediction,orig_prediction in zip(predict_labels,data.labels_val):
    ind1 = np.argmax(prediction)
    ind2= np.argmax(orig_prediction)
    predictions.append(ind1)
    orig_predictions.append(ind2)

cm=confusion_matrix(orig_predictions, predictions)

plot_confusion_matrix(cm, data.names_class)