# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 01:33:53 2019

Lyme Detector Disease

@author: Noman
"""
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
# Import adjustText, initialize list of texts
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import itertools
import re

from keras import models
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation,Dense,LSTM, Dropout, GaussianDropout, Flatten
from keras.layers import Conv1D,Conv2D, MaxPooling2D, MaxPooling1D
from keras import backend as K
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2

from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
#Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import natsort 
img_width, img_height = 64 , 64 
folderpath = "F:\lyme-disease\\processed_dataset\\"
folderpathaugmented = "F:\lyme-disease\\processed_dataset\\augmentation\\"
# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory(folderpath + "train\\", target_size=(img_width, img_height), color_mode='rgb', class_mode='categorical', batch_size=23, shuffle=True)
val_it = datagen.flow_from_directory(folderpath + "val\\", target_size=(img_width, img_height), color_mode='rgb', class_mode='categorical', batch_size=23, shuffle=True)
test_it = datagen.flow_from_directory(folderpath + "test\\", target_size=(img_width, img_height), color_mode='rgb', class_mode='categorical', batch_size=23, shuffle=True)


# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.2f, max=%.2f' % (batchX.shape, batchX.min(), batchX.max()))



"""
model.add(Conv2D(64, (6,6),strides=6, padding='valid',activation='tanh', kernel_regularizer=l2(reg)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (6,6),strides=6, padding='valid',activation='elu', kernel_regularizer=l2(reg)))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(64 , init="uniform", activation='sigmoid'))
"""

reg = 0.007
model = Sequential()
model.add(Conv2D(128, (3,3),strides=3, padding='valid',activation='tanh', input_shape=(img_width, img_height, 3), kernel_regularizer=l2(reg)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=3, padding='valid', data_format='channels_first'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(GaussianDropout(0.7))
model.add(Dense(128,  init="uniform", activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=3, padding='valid', data_format='channels_first'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#model.add(Dropout(0.5))
model.add(GaussianDropout(0.7))
#model.add(Dense(256,  init="uniform", activation='tanh'))
#model.add(Dense(128,  init="uniform", activation='tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(128,  init="uniform", activation='elu'))


#model.add(Dense(150 , init="uniform", activation='tanh'))
#model.add(Dense(100 , init="uniform", activation='tanh'))
model.add(Flatten()) 

model.add(Dense(2 ,activation='softplus'))
model.summary()

#adam = Keras.optimizer.Adam(lr= 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.0, amsgrad=False)
#model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, beta_1=0.7, beta_2=0.777, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0), metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer="RMSprop", metrics=['accuracy'])
csv_logger = CSVLogger('F:\lyme-disease\\training_log.csv', separator=',', append=False)
#history = model.fit(xTrain, yTrain, epochs=50, batch_size=6,shuffle = True,  validation_split=0.2, verbose=2, callbacks=[csv_logger] )
history = model.fit_generator(train_it, epochs=300, steps_per_epoch=2, validation_data=val_it, validation_steps=2)
#accuracy = model.evaluate(xTest, yTest, batch_size=6, verbose=0)
accuracy = model.evaluate_generator(val_it, steps=2)
print(accuracy)

log_data = pd.read_csv('F:\lyme-disease\\training_log.csv', sep=',', engine='python')
#print(log_data)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        folderpath + "test\\", #"F:\lyme-disease\\processed_dataset\\augmentation\\test\\",
        target_size=(img_width, img_height),
        color_mode="rgb",
        shuffle = True,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

test_generator.reset()


#model load
#model = load_model("F:\lyme-disease\\lyme_model.h5")
#model.summary()

Y_pred = model.predict_generator(test_generator, steps = nb_samples)
#print(Y_pred)


y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)


print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['Yes', 'No']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
target_names = ['Yes', 'No']



tn, fp, fn, tp = confusion_matrix(test_generator.classes, y_pred).ravel()
      


#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 


#Precision 
Precision = tp/(tp+fp) 


#Recall 
Recall = tp/(tp+fn) 



#F1 Score
f1 = (2*Precision*Recall)/(Precision + Recall)


#Specificity 
Specificity = tn/(tn+fp)


print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
print("Accuracy {:0.2f}%:".format(Accuracy))
print("Precision {:0.2f}".format(Precision))
print("Recall {:0.2f}".format(Recall))
print("F1 Score {:0.2f}".format(f1))
print("Specificity {:0.2f}".format(Specificity))



#labels = (train_it.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]

#print(predictions)


#metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
#metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred))


#print(Y_pred)
#print(val_it.classes)

"""
auc = roc_auc_score(test_generator.classes, Y_pred)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(test_generator.classes, Y_pred)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
"""


#save keras model
#model.save("F:\lyme-disease\\lyme_model.h5")
#print("Model Saved!")


