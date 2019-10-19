# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

cd /content/gdrive/My\ Drive/vip2

import numpy as np
import cv2
import pickle
import keras

import os
count = 0
f = 'NWPU-RESISC12'
for i in os.listdir('NWPU-RESISC12'):
  for j in os.listdir(os.path.join(f,i)):
    for k in os.listdir(os.path.join(f,i,j)):
      count += 1

print('count:',count)

train_dir = 'NWPU-RESISC12/train'
validation_dir = 'NWPU-RESISC12/test'
image_size = 224
nTrain = 6600
nVal = 1800

descriptors_train = np.load('descriptors_train.npy', allow_pickle=True)
descriptors_test = np.load('descriptors_test.npy', allow_pickle=True)

print('descriptors_train.shape:',descriptors_train.shape)
print('descriptors_test.shape:',descriptors_test.shape)

from scipy.cluster.vq import kmeans, vq

with open('codebook.pkl','rb') as handle:
    codebook = pickle.load(handle)
def build_histogram(des):
    k = 50
    code, distortion = vq(des, codebook)
    bowhist,bin_edge = np.histogram(code, k, density=True)
    return bowhist

temp = []
for i in descriptors_train[:,1]:
  temp.append(build_histogram(i))

des_train = np.array(temp)
print(des_train.shape)

temp2 = []
for i in descriptors_test[:,1]:
  temp2.append(build_histogram(i))
des_test = np.array(temp2)
print(des_test.shape)

import pickle
with open('VGG_Places_Bottleneck_Train_Features.pkl','rb') as handle:
    train_features = pickle.load(handle)
with open('VGG_Places_Bottleneck_Train_Labels.pkl','rb') as handle:
    train_labels = pickle.load(handle)
with open('VGG_Places_Bottleneck_Test_Features.pkl','rb') as handle:
    test_features = pickle.load(handle)
with open('VGG_Places_Bottleneck_Test_Labels.pkl','rb') as handle:
    test_labels = pickle.load(handle)

print('train_features.shape:',train_features.shape)
print('train_labels.shape:', train_labels.shape)
print('test_features.shape:',test_features.shape)
print('test_labels.shape:',test_labels.shape)

bottle_train = train_features
bottle_test = test_features

# aggregate train
agg_train_features = np.hstack((des_train,bottle_train))
agg_train_features.shape

# aggregate test
agg_validation_features = np.hstack((des_test,bottle_test))
agg_validation_features.shape

# aggregate label
agg_train_labels = descriptors_train[:,0].astype('int32')
agg_validation_labels = descriptors_test[:,0].astype('int32')

tttt = np.argmax(train_labels,axis=1)
tttt.shape
count = 0
for i in range(6600):
  if tttt[i] == d_label[i]:
    count += 1
print(count)

# DCNN
from keras import models
from keras import layers
from keras import optimizers
 
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=agg_train_features.shape[1]))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

batch_size = 20

history = model.fit(agg_train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(agg_validation_features,test_labels),
                    verbose=1
                    )

# evaluate the model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

predictions = model2.predict(agg_train_features,batch_size=batch_size)
predicted_classes = np.argmax(predictions,axis=1)
def scores(y_true,y_pred):
  accuracy = accuracy_score(y_true,y_pred)
  precision = precision_score(y_true,y_pred,average='macro')
  recall = recall_score(y_true,y_pred,average='macro')
  f1 = f1_score(y_true,y_pred,average='macro')
  confusionmatrix = confusion_matrix(y_true, y_pred)
  print("Accuracy:  {}".format(accuracy))
  print("Precision: {}".format(precision))
  print("Recall:    {}".format(recall))
  print("F1:        {}".format(f1))
  print("Confusion matrix:\n{}".format(confusionmatrix))

  scores(agg_train_labels,predicted_classes)