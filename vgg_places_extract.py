# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/')

cd /content/gdrive/My\ Drive/vip2

train_dir = 'NWPU-RESISC12/train'
validation_dir = 'NWPU-RESISC12/test'
image_size = 224
nTrain = 6600
nVal = 1800

import numpy as np
import pickle
from keras.models import load_model

vgg_places = load_model('vgg_places_30_epochs.h5')
vgg_places.pop()
vgg_places.pop()
vgg_places.pop()
print(vgg_places.summary())

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# extract train features
train_features = np.zeros(shape=(nTrain, 7* 7* 512))
train_labels = np.zeros(shape=(nTrain,12))
 
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_places.predict(inputs_batch)
    
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    print(i,end=' ')
    i += 1
    if i * batch_size >= nTrain:
        break

print('Done')

# extract validation features
validation_features = np.zeros(shape=(nVal, 7* 7* 512))
validation_labels = np.zeros(shape=(nVal,12))

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_places.predict(inputs_batch)

    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    print(i, end=' ')
    if i * batch_size >= nVal:
        break

print('done')

np.save('vgg_places_train_features',train_features)
np.save('vgg_places_train_labels',train_labels)
np.save('vgg_places_val_features',validation_features)
np.save('vgg_places_val_labels',validation_labels)