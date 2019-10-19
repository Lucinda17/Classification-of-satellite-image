# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/')

cd /content/gdrive/My\ Drive/vip2
import numpy as np

# load bottleneck features
train_features_places = np.load('vgg_places_train_features.npy')
train_features_da     = np.load('vgg_da_train_features.npy')
val_features_places  = np.load('vgg_places_val_features.npy')
val_features_da      = np.load('vgg_da_val_features.npy')

trian_label_places = np.load('vgg_places_train_labels.npy')
test_label_places  = np.load('vgg_places_val_labels.npy')

# concatenate features
train_features = np.hstack((train_features_places,train_features_da))
val_features = np.hstack((val_features_places, val_features_da))
train_labels = trian_label_places
val_labels = test_label_places

train_labels = np.argmax(train_labels, axis=1)

# train the model
from sklearn import svm
clf = svm.SVC(gamma='scale', kernel='linear')
clf.fit(train_features, train_labels)
predictions = clf.predict(val_features)
y_true = np.argmax(val_labels, axis=1)


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

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

scores(y_true,predictions)