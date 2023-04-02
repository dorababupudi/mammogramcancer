import os
import cv2
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

path = 'IRMADataset'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)
#definition of statisticals features extraction object
statistical_features = SelectKBest(score_func=f_classif, k=784)

for root, dirs, directory in os.walk(path): #read all images and insert into array
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j],0)
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            X_train.append(im2arr.ravel())
            Y_train.append(getID(name))
#convert array into numpy format        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)
#normalized the image data
X_train = X_train.astype('float32')
X_train = X_train/255
#shuffle the images data    
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
#now apply statistical features
fit = statistical_features.fit(X_train, Y_train)
#now extract important features by using statistical object
features = fit.transform(X_train)
print(features)
print(features.shape)

np.save('model/statistical_X.txt',features)
np.save('model/statistical_Y.txt',Y_train)

test = features[3]
test = test.reshape(28,28);
cv2.imshow("aa",cv2.resize(test,(200,200)))
cv2.waitKey(0)

