import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

train_path = 'train'
test_path  = 'test'

num_label = len(os.listdir(train_path))
label = os.listdir(train_path)
total_img = 0
for i in os.listdir(train_path):
    total_img += len(os.listdir(os.path.join(train_path, i)))

# Feature extraction
labels = []
features = []
count = 0
for folder in os.listdir(train_path):
    for image in os.listdir(os.path.join(train_path,folder)):
        filename = os.path.join(train_path,folder,image)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_des = 100
        threshold = 0.05
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
        kps, des = sift.detectAndCompute(gray, None)
        try:
            l = len(des)
        except:
            threshold = 0.01
            sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
            kps, des = sift.detectAndCompute(gray, None)
            l = len(des)
        while (min_des>l):
            threshold -= 0.001
            sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
            kps, des = sift.detectAndCompute(gray, None)
            l = len(des)
        
        labels.append(label.index(folder))
        features.append((kps,des))

# combine only descriptors
loc, des = list(zip(*features))
alldes = np.vstack(des)

# create cluster and codebook
import time
from scipy.cluster.vq import kmeans, vq

k = 50
alldes = np.float32(alldes)
e0 = time.time()
print('start kmeans...')
codebook, distortion = kmeans(alldes, k)
print('start vq...')
code, distortion = vq(alldes,codebook)
e1 = time.time()
print("Time: {}\nCluster: {}\nImages: {}".format(e1-e0,k,alldes.shape[0]))

# save and load codebook
import pickle
pickle.dump(codebook, open("codebook.pkl","wb"))
codebook = pickle.load(open("codebook.pkl","rb"))

# features normalization
# so that every image has the same number of features
label_bow = list()
bow = list()
pred = []
for i in range(6600):
    code, distortion = vq(features[i][1], codebook)
    pred.append(code)
    bowhist, bin_edges = np.histogram(code, k, density=True)
    label_bow.append([labels[i],bowhist])
    bow.append(bowhist)

# function to extract descriptors
def extract_features(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_des = 100
    threshold = 0.05
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
    kps, des = sift.detectAndCompute(gray, None)
    try:
        l = len(des)
    except:
        threshold = 0.01
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
        kps, des = sift.detectAndCompute(gray, None)
        l = len(des)
    while (min_des>l):
        #print(threshold,l)
        threshold -= 0.001
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=threshold)
        kps, des = sift.detectAndCompute(gray, None)
        l = len(des)
    return des

# fucntion to build histogram from descriptors
def build_histogram(des):
    k = 50
    code, distortion = vq(des, codebook)
    bowhist,bin_edge = np.histogram(code, k, density=True)
    return bowhist

# train NearestNeighbors and get prediction from model
from sklearn.neighbors import NearestNeighbors

y_pred = []
y_true = []
filenames = []
neighbor = NearestNeighbors(n_neighbors = 20)
neighbor.fit(bow)
for folder in os.listdir(test_path):
    print(folder)
    for image in os.listdir(os.path.join(test_path,folder)):
        filename = os.path.join(test_path,folder,image)
        descriptors = extract_features(filename)
        histogram = build_histogram(descriptors)
        dist, result = neighbor.kneighbors([histogram])
        result = result[0]
        result = [labels[r] for r in result]
        predicted_label = np.argmax(np.bincount(result))
        y_pred.append(predicted_label)
        y_true.append(label.index(folder))
        filenames.append(filename)

# evaluate the model
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

scores(y_true,y_pred)