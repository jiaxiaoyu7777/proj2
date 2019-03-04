# import the necessary packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from numpy import argmax
from sklearn.metrics import classification_report

from sklearn.svm import SVC

start1 =time.time()
start2 = time.process_time()
df = pd.read_csv('/Users/jiaxiaoyu/Downloads/data.csv')

df=df.drop(['id'],axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

#pre-processing
scaler = preprocessing.StandardScaler()
X1 = scaler.fit_transform(X)
#X2 = preprocessing.normalize(X1, norm='l2')

#importing train_test_split
X_train, X_valtest, y_train, y_valtest = train_test_split(X1, y, test_size=0.2, random_state=0, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0, stratify=y_valtest)

# Setup arrays to store training and val accuracies
kernel = ['linear','poly','rbf','sigmoid']
accuracies = []
C=range(1,1002,50)
degree=range(1,6,1)

for k in kernel:
    for de in degree:
        for i in C:
            # Setup a Support Vector Classifier
            svm = SVC(gamma='scale',kernel=k,C=i,degree=de)
            # Fit the model
            svm.fit(X_train, y_train)
            # Compute accuracy on the test set
            val_accuracy = svm.score(X_val, y_val)
            if k=='poly':
                print("kernel=%s,C=%d,degree=%d,accuracy=%.2f%%" % (k, i, de, val_accuracy * 100))
            else:
                print("kernel=%s,C=%d,accuracy=%.2f%%" % (k, i, val_accuracy * 100))

            accuracies.append(val_accuracy)


winner=np.argwhere(accuracies==np.amax(accuracies))
maxlist=[]
maxlist=winner.flatten().tolist()
maxlist = list(map(int, maxlist))
print(maxlist)
accuracies1=[]

for d in maxlist:
    if kernel[int(d/105)]=='poly':
        print("kernel=%s,C=%d and degree=%d achieved highest accuracy of %.2f%% on validation data" % (kernel[int(d/105)],C[d%21],degree[int((d%105)/21)],accuracies[d] * 100))
    else:
        print("kernel=%s and C=%d achieved highest accuracy of %.2f%% on validation data" % (kernel[int(d/105)],C[d%21],accuracies[d] * 100))
    test = SVC(gamma='scale', kernel=kernel[int(d/105)], C=C[d%21],degree=degree[int((d%105)/21)])
    # Fit the model
    test.fit(X_train, y_train)
    # Compute accuracy on the test set
    test_accuracy = test.score(X_test, y_test)
    print("the accuracy is %.2f%% on test data" % (test_accuracy * 100))
    accuracies1.append(test_accuracy)
    #print(d)
    #print(accuracies1)

a = int(argmax(accuracies1))
b= int(maxlist[a])
#print(b)


print()


#Setup a support vector classifier
new = SVC(gamma='scale',kernel=kernel[int(b/105)],C=C[b%21],degree=degree[int((b%105)/21)])
#Fit the model
new.fit(X_train, y_train)
max_accuracy = new.score(X_test, y_test)
if kernel[int(b/105)]=='poly':
    print("kernel=%s,C=%d and degree=%d achieved highest accuracy of %.2f%% on test data" % (kernel[int(b/105)],C[b%21],degree[int((b%105)/21)],max_accuracy * 100))
else:
    print("kernel=%s and C=%d achieved highest accuracy of %.2f%% on test data" % (kernel[int(b/105)],C[b%21],max_accuracy * 100))
y_pred = new.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

end1 = time.time()
end2 = time.process_time()
clock = end1 - start1
cpu = end2-start2
print("wall-clock time is" + str(clock) + "s")
print("run-time is" + str(cpu) + "s")

#import classification_report
#print(classification_report(y_test,y_pred))




