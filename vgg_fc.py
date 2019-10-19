import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from __future__ import print_function
from keras.applications import VGG16
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import keras

# instantiate base model
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

print(vgg_conv.summary())

# unzip the dataset
!unzip -q '/content/gdrive/My Drive/Colab Notebooks/vip2/NWPUvip.zip'

train_dir = '/content/NWPU-RESISC12/train'
validation_dir = '/content/NWPU-RESISC12/test'
image_size = 224
nTrain = 6600
nVal = 1800

# data generator 
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
 
# read train data from directory
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# save bottleneck features of train set
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,12))
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
         
# save the features extracted from the training dataset to be passed to model.fit for training        
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

# read validation data from directory
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# save bottlenect features of validation set
validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,12))
i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

# save the features extracted from the validation dataset to be passed to model.fit for training 
validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))


# Craete fully connected classifier
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

# train the model
history = model.fit(train_features,
                    train_labels,
                    epochs=50,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

# model evaluation 
# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

# Show the errors
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))
print('Accuracy')
print((nVal-len(errors))*100/nVal)
print(len(errors))

# save the model
model.save('/content/gdrive/My Drive/Colab Notebooks/vip2/vgg-fullyconnected.h5')

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,f1_score
print(confusion_matrix(ground_truth, predictions, labels=None, sample_weight=None))
print(precision_score(ground_truth, predictions,average='macro'))
print(recall_score(ground_truth, predictions,average='macro'))
print(accuracy_score(ground_truth, predictions))
print(f1_score(ground_truth, predictions,average='macro'))