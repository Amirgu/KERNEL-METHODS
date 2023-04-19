import numpy as np
import pickle

import csv
import time
from KernelFDA import KernelFDA
from kernels import *
from KernelSVM import *
# Load the data

with open('/home/amine/KERNELOARECONSTRUCTEDTRAINORMALTEST.pkl', 'rb') as f:
    Z = pickle.load(f)

with open('projet/data/training_labels.pkl', 'rb') as f:
    y = pickle.load(f)

    
eps=0.0001
Z=(Z-np.mean(Z,axis=0))/(np.std(Z,axis=0)+eps)
X_train=Z[:6000,:6000]
y_train=y
X_test=Z[6000:,:6000]


# Perform dimensionality reduction with KernelPCA using RBF kernel
ke=RBF(30.)
rbf_kernel_fda = KernelFDA(n_components=100, kernel=ke.kernel)

X_train_reduced = rbf_kernel_fda.fit_transform(X_train,y_train)

X_test_reduced = rbf_kernel_fda.transform(X_test)



y_train =np.array( [-1 if x == 0 else 1 for x in y_train])


#Classification
rbf=RBF(0.5)

rf = KernelSVC(C=1, kernel=rbf.kernel)

rf.fit(X_train_reduced,y_train,class_weights=[7,1])

y_pred = rf.predict(X_test_reduced)

y_pred=np.array( [0 if x == -1 else 1 for x in y_pred])

predictions=y_pred

rows = [{'Id': i+1, 'Predicted': prediction} for i, prediction in enumerate(predictions)]

with open('submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['Id', 'Predicted']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader() 
    writer.writerows(rows)