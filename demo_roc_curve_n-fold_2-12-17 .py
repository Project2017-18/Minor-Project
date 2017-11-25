import pandas as pd
import os
from collections import defaultdict
from itertools import cycle
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import matthews_corrcoef
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import low_variance
from sklearn.preprocessing import label_binarize
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy import interp
import time
import xlsxwriter
color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
datasets=[]
xlsxfile=[]
for i in os.listdir("dataset"):
    datasets.append(i)
    xlsxfile.append(str(i)[:-4]+'.xlsx')
model1 = SVC(kernel='poly',probability=True) 
model2 = RandomForestClassifier()
##model3 = GaussianNB()
##model4 = KNeighborsClassifier()
##model5 = DecisionTreeClassifier(random_state=0)


for data ,outfile in zip(datasets,xlsxfile):
    color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
    #getting the data
    mat = scipy.io.loadmat('dataset/'+data)
    X = mat['data']
    Y = X[:, 0]
    Y=Y.astype(int)
    print(Y)
    X=X[:,1:]
    print(X)
    a=np.unique(Y)
    Y1 = label_binarize(Y, classes=a.tolist())
    print(Y1)
    n_classes = a.size
    colors = cycle(color[0:n_classes])
    n_samples, n_features = X.shape
    
#to take data from datasets
    #n value depends on no of rows of class label which is present minimum no of times
    
#to find class having minimum no of times ex: 70 classes 4 no of times. if class 4 has arrives less time, n(1) is printed        
    d = defaultdict(int)
    for i in Y:
        d[i] += 1
    n = min(d.iteritems(), key=lambda x: x[1])
    print (n[1])

    
    for model, label,sheet in zip([ model1,model2], [ 'SVM','Random_Forest'],['sheet1','sheet2']):
        
        start=time.time()
        accuracy_list=[]
        precision_list=[]
        recall_list=[]
        f1_list=[]
        feature_list=[]
        nfeature=[]
        classifier_list=[]
        time_list=[]
        reliefF_acc_list=[]
        #matthews_corrcoef_list=[]
        
        classifier=label
        
        classifier1 = OneVsRestClassifier(model)

        
        features_list=[10,20,40,60,80,100]
        for i in features_list:
            

            
            #ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            skf = StratifiedKFold(n_splits=n[1],shuffle=True)
#to find the number of folds
            #loo = LeaveOneOut()
            accuracy=0
            precision=0
            recall=0
            f1score=0
            #mcc=0
            y_train=[]
            y_test=[]
            X_train=[]
            X_test=[]
#fpr tpr will come to every class. 
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for j in range(n_classes):
                fpr[j]=[]
#putting every value of fpr in 2d array in every fold for every class
                tpr[j]=[]
                roc_auc[j]=[]
            mean_fpr = np.linspace(0, 1, 100) 
#mean of every value is found and roc is found  
            for train, test in skf.split(X,Y):
            
                y_train = label_binarize(Y[train], classes=a.tolist())
                y_test = label_binarize(Y[test], classes=a.tolist())
                A=classifier1.fit(X[train], y_train)
                y_score = A.predict_proba(X[test])
                
                # Compute ROC curve and ROC area for each class
                
                for j in range(n_classes):
                    fpr1, tpr1, _ = roc_curve(y_test[:, j], y_score[:, j])
                    
                    tpr[j].append(interp(mean_fpr, fpr1, tpr1))
#graph for fpr, tpr. 
                    roc_auc1 = auc(fpr1, tpr1)
                    roc_auc[j].append(roc_auc1)


            lw=2

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)

            
            for j, color in zip(range(n_classes), colors):
                plt.plot(mean_fpr, np.mean(tpr[j],axis=0), color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(j, np.mean(roc_auc[j], axis=0)))

            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()
            

            
       


