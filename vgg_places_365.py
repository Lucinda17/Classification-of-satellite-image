# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/')

cd /content/gdrive/My\ Drive/vip2

!unzip -q "drive/My Drive/vip2/NWPUvip.zip"

train_dir = 'NWPU-RESISC12/train'
test_dir  = 'NWPU-RESISC12/test' 
image_size = 224
nTrain = 6600
nTest = 1800

cd drive/My\ Drive/vip2
# load vgg16_places365
from vgg16_places_365 import VGG16_Places365
vgg16_places = VGG16_Places365(weights='places', include_top=False, input_shape=(224, 224, 3))


from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np

# Train :: Extract Features
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,12))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg16_places.predict(inputs_batch)
    train_features[i*batch_size:(i+1)*batch_size]=features_batch
    train_labels[i*batch_size:(i+1)*batch_size]=labels_batch
    i+=1
    print(i,end='')
    if i*batch_size >= nTrain:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))
print('\nDONE')

# Save bottleneck features
import pickle
with open('drive/My Drive/vip2/VGG_Places_Bottleneck_Train_Features.pkl','wb') as handle:
    pickle.dump(train_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('drive/My Drive/vip2/VGG_Places_Bottleneck_Train_Labels.pkl','wb') as handle:
    pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Extract test
test_features = np.zeros(shape=(nTest, 7, 7, 512))
test_labels = np.zeros(shape=(nTest,12))

print('Data generator')
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

print('save features')
i = 0
for inputs_batch, labels_batch in test_generator:
    features_batch = vgg16_places.predict(inputs_batch)
    test_features[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    print(i,end=' ')
    if i * batch_size >= nTest:
        break

test_features = np.reshape(test_features, (nTest, 7 * 7 * 512))
# Save bottleneck features
import pickle
with open('drive/My Drive/vip2/VGG_Places_Bottleneck_Test_Features.pkl','wb') as handle:
    pickle.dump(test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('drive/My Drive/vip2/VGG_Places_Bottleneck_Test_Labels.pkl','wb') as handle:
    pickle.dump(test_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

from keras import models
from keras import layers
from keras import optimizers

# Freeze all the layers
for layer in vgg16_places.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg16_places.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg16_places)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))
from keras.optimizers import Adam
validation_generator = test_generator

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

import matplotlib.pyplot as plt
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

# evaluate the model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
y_true = test_generator.classes
predictions = model.predict_generator(test_generator)
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

scores(y_true,predicted_classes)



# data augmentation
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import keras
from keras import optimizers


# Specify the data augmentation techniques in the trainig image data generator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
validation_datagen = ImageDataGenerator(rescale=1./255)

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
vgg_places.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the Model
# multiplied the steps_per_epoch by 2 for data augmentation.
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = vgg_places.fit_generator(
      train_generator,
      steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
      epochs=40,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      callbacks=[es_callback],
      verbose=1
      )

# Save the Model
vgg_places.save('vgg_places_data_aug.h5')
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
predictions = model.predict_generator(validation_generator)
predicted_classes = np.argmax(predictions,axis=1)
y_true = validation_generator.classes
scores(y_true,predicted_classes)
