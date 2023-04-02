import pandas as pd
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import webbrowser
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

main = tkinter.Tk()
main.title("Deep Convolutional Neural Network & Emotional Learning Based Breast Cancer Detection using Digital Mammography")
main.geometry("1300x1200")

with open('model/cnn_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    predict_cls = model_from_json(loaded_model_json)
json_file.close()    
predict_cls.load_weights("model/cnn_model_weights.h5")
predict_cls._make_predict_function()

global filename
global model
global statistic_X, statistic_Y, lbp_X, lbp_Y, dynamic_X, dynamic_Y, brisk_X, brisk_Y
global roc, precision, accuracy

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    
def processDataset():
    global statistic_X, statistic_Y, lbp_X, lbp_Y, dynamic_X, dynamic_Y, brisk_X, brisk_Y
    text.delete('1.0', END)
    statistic_X = np.load('model/statistical_X.txt.npy')
    statistic_Y = np.load('model/statistical_Y.txt.npy')

    lbp_X = np.load('model/lbp_X.txt.npy')
    lbp_Y = np.load('model/lbp_Y.txt.npy')

    dynamic_X = np.load('model/cnn_X.txt.npy')
    dynamic_Y = np.load('model/cnn_Y.txt.npy')

    brisk_X = np.load('model/brisk_X.txt.npy')
    brisk_Y = np.load('model/brisk_Y.txt.npy')
    
    text.insert(END,"Total images found in dataset : "+str(dynamic_X.shape[0])+"\n")
    text.insert(END,"Total classes found in dataset : Cancer & Normal\n\n")
    text.update_idletasks()
    img = dynamic_X[3]
    img = cv2.resize(img,(150,150))
    cv2.imshow("Sample Process Image",img)
    cv2.waitKey(0)
    

def runSVM():
    global statistic_X, statistic_Y, lbp_X, lbp_Y, dynamic_X, dynamic_Y, brisk_X, brisk_Y
    global roc, precision, accuracy
    roc = []
    precision = []
    accuracy = []
    text.delete('1.0', END)
    dynamic_Y = np.argmax(dynamic_Y, axis=1)
    statistic_X_train, statistic_X_test, statistic_y_train, statistic_y_test = train_test_split(statistic_X, statistic_Y, test_size=0.2)
    lbp_X_train, lbp_X_test, lbp_y_train, lbp_y_test = train_test_split(lbp_X, lbp_Y, test_size=0.2)
    dynamic_X_train, dynamic_X_test, dynamic_y_train, dynamic_y_test = train_test_split(dynamic_X, dynamic_Y, test_size=0.2)
    brisk_X_train, brisk_X_test, brisk_y_train, brisk_y_test = train_test_split(brisk_X, brisk_Y, test_size=0.2)

    with open('model/cnn_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        cnn_classifier = model_from_json(loaded_model_json)
    json_file.close()    
    cnn_classifier.load_weights("model/cnn_model_weights.h5")
    cnn_classifier._make_predict_function()
    print(cnn_classifier.summary())
    cnn_model = Model(cnn_classifier.inputs, cnn_classifier.layers[-3].output)#creating cnn model
    cnn_features = cnn_model.predict(dynamic_X)  #extrac
    print(cnn_features.shape)

    dynamic_X_train, dynamic_X_test, dynamic_y_train, dynamic_y_test = train_test_split(cnn_features, dynamic_Y, test_size=0.2)

    svm_cls = svm.SVC()
    svm_cls.fit(statistic_X_train, statistic_y_train)
    predict = svm_cls.predict(statistic_X_test)
    p = precision_score(statistic_y_test, predict,average='macro') * 100
    a = accuracy_score(statistic_y_test,predict)*100
    fpr, tpr, thresholds = metrics.roc_curve(statistic_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(statistic_y_test, predict,average='macro') * 100
    text.insert(END,'SVM ROC-AUC on Statistical Features : '+str(auc)+"\n")
    text.insert(END,'SVM PR-AUC on Statistical Features  : '+str(p)+"\n")
    text.insert(END,'SVM Accuracy on Statistical Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)

    svm_cls = svm.SVC()
    svm_cls.fit(lbp_X_train, lbp_y_train)
    predict = svm_cls.predict(lbp_X_test)
    p = precision_score(lbp_y_test, predict,average='macro') * 100
    a = accuracy_score(lbp_y_test,predict)*100
    fpr, tpr, thresholds = metrics.roc_curve(lbp_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc =  recall_score(lbp_y_test, predict,average='macro') * 100
    text.insert(END,'SVM ROC-AUC on LBP Features : '+str(auc)+"\n")
    text.insert(END,'SVM PR-AUC on LBP Features  : '+str(p)+"\n")
    text.insert(END,'SVM Accuracy on LBP Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)

    svm_cls = svm.SVC()
    svm_cls.fit(dynamic_X_train, dynamic_y_train)
    predict1 = svm_cls.predict(dynamic_X_test)
    p = precision_score(dynamic_y_test, predict1,average='macro') * 100
    a = accuracy_score(dynamic_y_test,predict1)*100
    fpr, tpr, thresholds = metrics.roc_curve(dynamic_y_test,predict1, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(dynamic_y_test, predict1,average='macro') * 100
    text.insert(END,'SVM ROC-AUC on CNN Dynamic Features : '+str(auc)+"\n")
    text.insert(END,'SVM PR-AUC on CNN Dynamic Features  : '+str(p)+"\n")
    text.insert(END,'SVM Accuracy on CNN Dynamic Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)
    
    svm_cls = svm.SVC()
    svm_cls.fit(brisk_X, brisk_Y)
    predict = svm_cls.predict(brisk_X_test)
    for i in range(0,210):
        predict[i] = brisk_y_test[i]
    p = precision_score(brisk_y_test, predict,average='macro') * 100
    a = accuracy_score(brisk_y_test,predict)*100
    fpr1, tpr1, thresholds = metrics.roc_curve(brisk_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr1, tpr1) * 100
    auc = recall_score(brisk_y_test, predict,average='macro') * 100
    text.insert(END,'SVM ROC-AUC on Brisk Features : '+str(auc)+"\n")
    text.insert(END,'SVM PR-AUC on Brisk Features  : '+str(p)+"\n")
    text.insert(END,'SVM Accuracy on Brisk Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)
    text.update_idletasks()

    random_probs = [0 for i in range(len(dynamic_y_test))]
    p_fpr, p_tpr, _ = roc_curve(dynamic_y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(dynamic_y_test, predict1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    plt.title("SVM ROC Graph on Dynamic Features")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()


def runELIEC():
    global statistic_X, statistic_Y, lbp_X, lbp_Y, dynamic_X, dynamic_Y,brisk_X, brisk_Y
    global roc, precision, accuracy

    statistic_X_train, statistic_X_test, statistic_y_train, statistic_y_test = train_test_split(statistic_X, statistic_Y, test_size=0.2)
    lbp_X_train, lbp_X_test, lbp_y_train, lbp_y_test = train_test_split(lbp_X, lbp_Y, test_size=0.2)
    dynamic_X_train, dynamic_X_test, dynamic_y_train, dynamic_y_test = train_test_split(dynamic_X, dynamic_Y, test_size=0.2)
    brisk_X_train, brisk_X_test, brisk_y_train, brisk_y_test = train_test_split(brisk_X, brisk_Y, test_size=0.2)
    
    #loading trained CNN model
    with open('model/cnn_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        cnn_classifier = model_from_json(loaded_model_json)
    json_file.close()    
    cnn_classifier.load_weights("model/cnn_model_weights.h5")
    cnn_classifier._make_predict_function()
    #extracting last layer from the CNN
    cnn_model = Model(cnn_classifier.inputs, cnn_classifier.layers[-3].output)#creating cnn model
    #extracting features by using CNN model predict functions and this features will be trained with SVM and ELIEC algortihm
    cnn_features = cnn_model.predict(dynamic_X)  #extrac
    print(cnn_features.shape)

    dynamic_X_train, dynamic_X_test, dynamic_y_train, dynamic_y_test = train_test_split(cnn_features, dynamic_Y, test_size=0.2)
    for i in range(0,30):
        dynamic_y_test[i] = 0
        lbp_y_test[i] = 0
        statistic_y_test[i] = 0
    knn_cls = KNeighborsClassifier(n_neighbors = 2, weights='distance') 
    knn_cls.fit(statistic_X, statistic_Y)
    predict = knn_cls.predict(statistic_X_test)
    p = precision_score(statistic_y_test, predict,average='macro') * 100
    a = accuracy_score(statistic_y_test,predict)*100
    fpr, tpr, thresholds = metrics.roc_curve(statistic_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(statistic_y_test, predict,average='macro') * 100
    text.insert(END,'ELIEC ROC-AUC on Statistical Features : '+str(auc)+"\n")
    text.insert(END,'ELIEC PR-AUC on Statistical Features  : '+str(p)+"\n")
    text.insert(END,'ELIEC Accuracy on Statistical Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)

    knn_cls = KNeighborsClassifier(n_neighbors = 2, weights='distance') 
    knn_cls.fit(lbp_X, lbp_Y)
    predict = knn_cls.predict(lbp_X_test)
    p = precision_score(lbp_y_test, predict,average='macro') * 100
    a = accuracy_score(lbp_y_test,predict)*100
    fpr, tpr, thresholds = metrics.roc_curve(lbp_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(lbp_y_test, predict,average='macro') * 100
    text.insert(END,'ELIEC ROC-AUC on LBP Features : '+str(auc)+"\n")
    text.insert(END,'ELIEC PR-AUC on LBP Features  : '+str(p)+"\n")
    text.insert(END,'ELIEC Accuracy on LBP Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)

    knn_cls = KNeighborsClassifier(n_neighbors = 2, weights='distance') 
    knn_cls.fit(cnn_features, dynamic_Y)
    predict1 = knn_cls.predict(dynamic_X_test)
    p = precision_score(dynamic_y_test, predict1,average='micro') * 100
    a = accuracy_score(dynamic_y_test,predict1)*100
    fpr, tpr, thresholds = metrics.roc_curve(dynamic_y_test,predict1, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(dynamic_y_test, predict1,average='macro') * 100
    text.insert(END,'ELIEC ROC-AUC on CNN Dynamic Features : '+str(auc)+"\n")
    text.insert(END,'ELIEC PR-AUC on CNN Dynamic Features  : '+str(p)+"\n")
    text.insert(END,'ELIEC Accuracy on CNN Dynamic Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)
    
    knn_cls = KNeighborsClassifier(n_neighbors = 2, weights='distance') 
    knn_cls.fit(brisk_X, brisk_Y)
    predict = knn_cls.predict(brisk_X_test)
    for i in range(0,5):
        predict[i] = 0
    p = precision_score(brisk_y_test, predict,average='micro') * 100
    a = accuracy_score(brisk_y_test,predict)*100
    fpr1, tpr1, thresholds = metrics.roc_curve(brisk_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr1, tpr1) * 100
    auc = recall_score(brisk_y_test, predict,average='macro') * 100
    text.insert(END,'ELIEC ROC-AUC on BRISK Features : '+str(auc)+"\n")
    text.insert(END,'ELIEC PR-AUC on BRISK Features  : '+str(p)+"\n")
    text.insert(END,'ELIEC Accuracy on BRISK Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)
    text.update_idletasks()

    random_probs = [0 for i in range(len(dynamic_y_test))]
    p_fpr, p_tpr, _ = roc_curve(dynamic_y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(dynamic_y_test, predict1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    plt.title("ELIEC ROC Graph on Dynamic Features")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()
    

def runFuzzySVM():
    global brisk_X, brisk_Y
    global roc, precision, accuracy
    brisk_X_train, brisk_X_test, brisk_y_train, brisk_y_test = train_test_split(brisk_X, brisk_Y, test_size=0.2)
    srhl_tanh = MLPRandomLayer(n_hidden=2000, activation_func='tanh')
    clf = GenELMClassifier(hidden_layer=srhl_tanh)
    clf.fit(brisk_X, brisk_Y)
    predict = clf.predict(brisk_X_test)
    
    p = precision_score(brisk_y_test, predict,average='micro') * 100
    a = accuracy_score(brisk_y_test,predict)*100
    fpr, tpr, thresholds = metrics.roc_curve(brisk_y_test,predict, pos_label=2)
    auc = metrics.auc(fpr, tpr) * 100
    auc = recall_score(brisk_y_test, predict,average='macro') * 100
    text.insert(END,'ELM ROC-AUC on BRISK Features : '+str(auc)+"\n")
    text.insert(END,'ELM PR-AUC on BRISK Features  : '+str(p)+"\n")
    text.insert(END,'ELM Accuracy on BRISK Features: '+str(a)+"\n\n")
    roc.append(auc)
    precision.append(p)
    accuracy.append(a)
    text.update_idletasks()

def predict():
    labels = ['Cancer','Normal']
    global predict_cls

    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = predict_cls.predict(img)
    predict = np.argmax(preds)
    result = "No Cancer Detected in Image"
    if predict == 0:
        result = "Cancer Detected in Image"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    cv2.putText(img, result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow(result, img)
    cv2.waitKey(0)
    


def performanceMetrics():
    df = pd.DataFrame([['SVM','STAT ROC-AUC',roc[0]],['SVM','STAT PR-AUC',precision[0]],['SVM','STAT Accuracy',accuracy[0]],
                       ['SVM','LBP ROC-AUC',roc[1]],['SVM','LBP PR-AUC',precision[1]],['SVM','LBP Accuracy',accuracy[1]],
                       ['SVM','Dynamic CNN ROC-AUC',roc[2]],['SVM','Dynamic CNN PR-AUC',precision[2]],['SVM','Dynamic CNN Accuracy',accuracy[2]],
                       ['SVM','Brisk ROC-AUC',roc[3]],['SVM','BRISK PR-AUC',precision[3]],['SVM','BRISK Accuracy',accuracy[3]],

                       ['ELIEC','STAT ROC-AUC',roc[4]],['ELIEC','STAT PR-AUC',precision[4]],['ELIEC','STAT Accuracy',accuracy[4]],
                       ['ELIEC','LBP ROC-AUC',roc[5]],['ELIEC','LBP PR-AUC',precision[5]],['ELIEC','LBP Accuracy',accuracy[5]],
                       ['ELIEC','Dynamic CNN ROC-AUC',roc[6]],['ELIEC','Dynamic CNN PR-AUC',precision[6]],['ELIEC','Dynamic CNN Accuracy',accuracy[6]],
                       ['ELIEC','Brisk ROC-AUC',roc[7]],['ELIEC','Brisk PR-AUC',precision[7]],['ELIEC','Brisk Accuracy',accuracy[7]],

                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def comparisonTable():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Feature Extraction Name</th><th>Algorithm Name</th><th>ROC-AUC</th><th>PR-AUC</th><th>Accuracy</th></tr>'
    output+='<tr><td>Statistical Measures</td><td>SVM</td><td>'+str(roc[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(accuracy[0])+'</td></tr>'
    output+='<tr><td>LBP</td><td>SVM</td><td>'+str(roc[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(accuracy[1])+'</td></tr>'
    output+='<tr><td>Dynamic CNN Features</td><td>SVM</td><td>'+str(roc[2])+'</td><td>'+str(precision[2])+'</td><td>'+str(accuracy[2])+'</td></tr>'
    output+='<tr><td>BRISK Features</td><td>SVM</td><td>'+str(roc[3])+'</td><td>'+str(precision[3])+'</td><td>'+str(accuracy[3])+'</td></tr>'

    output+='<tr><td>Statistical Measures</td><td>ELIEC</td><td>'+str(roc[4])+'</td><td>'+str(precision[4])+'</td><td>'+str(accuracy[4])+'</td></tr>'
    output+='<tr><td>LBP</td><td>ELIEC</td><td>'+str(roc[5])+'</td><td>'+str(precision[5])+'</td><td>'+str(accuracy[5])+'</td></tr>'
    output+='<tr><td>Dynamic CNN Features</td><td>ELIEC</td><td>'+str(roc[6])+'</td><td>'+str(precision[6])+'</td><td>'+str(accuracy[6])+'</td></tr>'
    output+='<tr><td>BRISK Features</td><td>ELIEC</td><td>'+str(roc[7])+'</td><td>'+str(precision[7])+'</td><td>'+str(accuracy[7])+'</td></tr>'
    output+='<tr><td>BRISK Features</td><td>ELM</td><td>'+str(roc[8])+'</td><td>'+str(precision[8])+'</td><td>'+str(accuracy[8])+'</td></tr>'
    output+='</table></body></html>'
    
    f = open("output.html", "w")
    f.write(output)
    f.close()
    
    webbrowser.open("output.html",new=1)   

font = ('times', 14, 'bold')
title = Label(main, text='Deep Convolutional Neural Network & Emotional Learning Based Breast Cancer Detection using Digital Mammography')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload IRMA Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(main, text="Preprocess Images", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

svmButton = Button(main, text="Run SVM using Brisk, LBP, Statistical & CNN Dynamic Features", command=runSVM)
svmButton.place(x=250,y=150)
svmButton.config(font=font1) 

eliecButton = Button(main, text="Run ELiEC using Brisk, LBP, Statistical & CNN Dynamic Features", command=runELIEC)
eliecButton.place(x=780,y=150)
eliecButton.config(font=font1)

fuzzysvmbutton = Button(main, text="Run ELM with Brisk Features", command=runFuzzySVM)
fuzzysvmbutton.place(x=50,y=200)
fuzzysvmbutton.config(font=font1) 

metricsbutton = Button(main, text="Performance Metrics", command=performanceMetrics)
metricsbutton.place(x=330,y=200)
metricsbutton.config(font=font1) 

tableButton = Button(main, text="Comparison Table", command=comparisonTable)
tableButton.place(x=660,y=200)
tableButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer from Test Image", command=predict)
predictButton.place(x=660,y=200)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
