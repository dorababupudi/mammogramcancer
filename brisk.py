import os
import cv2
import numpy as np


path = 'IRMADataset'

labels = []
X_train = []
Y_train = []

BRISK = cv2.BRISK_create()#Brisk definition here=================================

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

error = ['929_A_0337_1.LEFT_MLO.LJPEG.1_highpass.png','458_C_0086_1.LEFT_MLO.LJPEG.1_highpass.png']

for root, dirs, directory in os.walk(path): #read all images and insert into array
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            if directory[j] not in error:
                img = cv2.imread(root+"/"+directory[j],0)
                img = cv2.resize(img, (256,256))
                keypoints, descriptors = BRISK.detectAndCompute(img, None) #now compute brisk features from image======================
                im2arr = np.array(descriptors)
                im2arr = cv2.resize(im2arr,(32,32))
                print(str(im2arr.shape))
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

print(X_train.shape)
print(Y_train.shape)
print(Y_train)
np.save('model/brisk_X.txt',X_train)
np.save('model/brisk_Y.txt',Y_train)

test = features[3]
test = test.reshape(32,32);
cv2.imshow("aa",cv2.resize(test,(200,200)))
cv2.waitKey(0)

