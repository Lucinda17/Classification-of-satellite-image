import cv2
import numpy as np

from keras.applications import VGG16
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import keras


vgg_conv = VGG16(weights='imagenet',
                 include_top=False,
                input_shape=(224,224,3))

# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/')
cd /content/gdrive/My\ Drive/vip2

train_dir = 'NWPU-RESISC12/train'
validation_dir = 'NWPU-RESISC12/test'
image_size = 224
nTrain = 6600
nTest = 1800

# data generator
from keras.preprocessing.image import ImageDataGenerator
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
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i*batch_size:(i+1)*batch_size]=features_batch
    train_labels[i*batch_size:(i+1)*batch_size]=labels_batch
    i+=1
    print(i,end=' ')
    if i*batch_size >= nTrain:
        break
print('\nDone')
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

# Extract test feature
test_features = np.zeros(shape=(nTest, 7, 7, 512))
test_labels = np.zeros(shape=(nTest,12))

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in test_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    test_features[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTest:
        break

test_features = np.reshape(test_features, (nTest, 7 * 7 * 512))


y_true.append(np.argmax(train_labels, axis=1))

# SVM
from sklearn import svm
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(train_features, y_true)

prediction = clf.predict(test_features)
y_true.append(np.argmax(test_labels, axis=1))

# Evaluate the model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

accuracy = accuracy_score(test_true,test_pred)
precision = precision_score(test_true,test_pred,average='micro')
recall = recall_score(test_true,test_pred,average='micro')
f1 = f1_score(test_true,test_pred,average='micro')
cm = confusion_matrix(test_true,test_pred)
print("Accuracy:  {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall:    {}".format(recall))
print("F1:        {}".format(f1))
print(cm)