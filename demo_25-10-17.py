import pandas as pd
import os
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
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
from skfeature.function.statistical_based import t_score
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
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
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    colors = cycle(color[0:n_classes])

    
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
        
        classifier=label
        table=pd.DataFrame()
        kfold = model_selection.KFold(n_splits=10,random_state=15)
        results = model_selection.cross_val_score(model,X,Y,cv=kfold)
        print(results)
        X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.5)    
        A=model.fit(X_train, y_train)
        prediction=A.predict(X_test)
        print("with out Any feature selection")
        print(label+" Accuracy: %.3f%%  "% (results.mean()*100.0))
        precision, recall, fscore, support = score(y_test, prediction,average='micro')
        accuracy_list.append(float(results.mean()))
        precision_list.append(float( precision))
        recall_list.append(float(recall))
        f1_list.append(float(fscore))
        time_list.append(float(time.time()-start))
        start=time.time()
        print(float(results.mean()))
        print(float( precision))
        print(float(recall))
        print(float(fscore))
        nfeature.append(X.shape[1])
        feature_list.append("with out Any feature selection")
        classifier_list.append(classifier)
        
        classifier1 = OneVsRestClassifier(model)

        reliefF_acc_list=[]
        features_list=[20,40,60,80,100]
        for i in features_list:
            
              #information_theoretical_based
##            print("CMIM feature selection algorithm ")
##            CMIM_sf,a,b=CMIM.cmim(X,Y,n_selected_features=i)
##            Xtemp=X[:,CMIM_sf[0:(i+1)]]
##            print(Xtemp)
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("CMIM")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_CMIM_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##            print(data)
##            print(i)
##
##            print("MRMR feature selection algorithm ")
##            mRMR_sf,a,b=MRMR.mrmr(X,Y,n_selected_features=i)
##            Xtemp=X[:,mRMR_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.9,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("MRMR")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.9,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_MRMR_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##            print(data)
##            print(i)

##            print("ICAP feature selection algorithm")
##            ICAP_sf,a,b=ICAP.icap(X,Y,n_selected_features=i)
##            Xtemp=X[:,ICAP_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("ICAP")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_ICAP_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("DISR feature selection algorithm ")
##            DISR_sf,a,b=DISR.disr(X,Y,n_selected_features=i)
##            Xtemp=X[:,DISR_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("DISR")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_DISR_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("MIM feature selection algorithm ")
##            MIM_sf,a,b=MIM.mim(X,Y,n_selected_features=i)
##            Xtemp=X[:,MIM_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("MIM")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_MIM_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("MIFS feature selection algorithm ")
##            MIFS_sf,a,b=MIFS.mifs(X,Y,n_selected_features=i)
##            Xtemp=X[:,MIFS_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("MIFS")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_MIFS_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("JMI feature selection algorithm ")
##            JMI_sf,a,b=JMI.jmi(X,Y,n_selected_features=i)
##            Xtemp=X[:,JMI_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("JMI")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_JMI_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("CIFE feature selection algorithm ")
##            CIFE_sf,a,b=CIFE.cife(X,Y,n_selected_features=i)
##            Xtemp=X[:,CIFE_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("CIFE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_CIFE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##            
##                      
##
##            print("LCSI feature selection algorithm ")
##            kwargs = {'function_name': 'JMI','n_selected_features':i}
##            LCSI_sf,a,b=LCSI.lcsi(X,Y,**kwargs)
##            Xtemp=X[:,LCSI_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("LCSI")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_LCSI_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            #similarity_based
##            print("FISHER_SCORE feature selection algorithm ")
##            score1 = fisher_score.fisher_score(X, Y)
##            idx = fisher_score.feature_ranking(score1)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("FISHER SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_FISHER_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
            print("reliefF feature selection algorithm ")
            score1 = reliefF.reliefF(X, Y)
            idx = reliefF.feature_ranking(score1)
            Xtemp=X[:,idx[0:(i+1)]]

            kfold = model_selection.KFold(n_splits=10)
            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)

            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
            B=model.fit(X_train, y_train)
            prediction=B.predict(X_test)
            precision, recall, fscore, support = score(y_test, prediction,average='micro')
            accuracy_list.append(float(results.mean()))
            reliefF_acc_list.append(1-float(results.mean()))
            precision_list.append(float( precision))
            recall_list.append(float(recall))
            f1_list.append(float(fscore))
            time_list.append(float(time.time()-start))
            start=time.time()
            print(float(results.mean()))
            print(float( precision))
            print(float(recall))
            print(float(fscore))
            feature_list.append("reliefF")
            nfeature.append(i)
            classifier_list.append(classifier)
            
            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
            A=classifier1.fit(X_train, y_train)
            y_score = A.predict_proba(X_test)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for j in range(n_classes):
                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
                roc_auc[j] = auc(fpr[j], tpr[j])

            lw=2

            
            for j, color in zip(range(n_classes), colors):
                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(j, roc_auc[j]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(str('ROC_reliefF_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
            plt.gcf().clear()

##            print("SPEC feature selection algorithm ")
##            kwargs = {'style': 0}
##            score1 = SPEC.spec(X, **kwargs)
##            idx = SPEC.feature_ranking(score1, **kwargs)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("SPEC")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_SPEC_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("TRACE_RATIO feature selection algorithm ")
##            idx, feature_score, subset_score = trace_ratio.trace_ratio(X, Y, i, style='fisher')
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("TRACE RATIO")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_TRACE_RATIO_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("LAP_SCORE feature selection algorithm ")
##            kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
##            W = construct_W.construct_W(X, **kwargs_W)
##            score1 = lap_score.lap_score(X, W=W)
##            idx = lap_score.feature_ranking(score1)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("LAP SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_LAP_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##            '''
##            #statistical_based
##            print("CFS feature selection algorithm ")
##            CFS_sf=CFS.cfs(X,Y)
##            Xtemp=X[:,CFS_sf[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("CFS")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_CFS_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##            '''
##            print("CHI_SCORE feature selection algorithm ")
##            score1=chi_square.chi_square(X,Y)
##            idx = chi_square.feature_ranking(score1)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("CHI SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_CHI_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("F_SCORE feature selection algorithm ")
##            score1 = f_score.f_score(X, Y)
##            idx = f_score.feature_ranking(score1)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("F SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_F_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("GINI_INDEX feature selection algorithm ")
##            score1 = gini_index.gini_index(X,Y)
##            idx = gini_index.feature_ranking(score1)
##            Xtemp=X[:,idx[0:(i+1)]]
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("GINI INDEX")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_GINI_INDEX_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
##
##            print("LOW_VARIANCE feature selection algorithm ")
##            p = 0.1 
##            selected_features = low_variance.low_variance_feature_selection(X, p*(1-p))
##            Xtemp=selected_features
##
##            kfold = model_selection.KFold(n_splits=10)
##            results = model_selection.cross_val_score(model,Xtemp,Y,cv=kfold)
##
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y,train_size=0.5,random_state=25)
##            B=model.fit(X_train, y_train)
##            prediction=B.predict(X_test)
##            precision, recall, fscore, support = score(y_test, prediction,average='micro')
##            accuracy_list.append(float(results.mean()))
##            precision_list.append(float( precision))
##            recall_list.append(float(recall))
##            f1_list.append(float(fscore))
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            print(float(results.mean()))
##            print(float( precision))
##            print(float(recall))
##            print(float(fscore))
##            feature_list.append("LOW VARIANCE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
##            A=classifier1.fit(X_train, y_train)
##            y_score = A.predict_proba(X_test)
##            
##            # Compute ROC curve and ROC area for each class
##            fpr = dict()
##            tpr = dict()
##            roc_auc = dict()
##            for j in range(n_classes):
##                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
##                roc_auc[j] = auc(fpr[j], tpr[j])
##
##            lw=2
##
##            
##            for j, color in zip(range(n_classes), colors):
##                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
##                         label='ROC curve of class {0} (area = {1:0.2f})'
##                         ''.format(j, roc_auc[j]))
##
##            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##            plt.xlim([0.0, 1.0])
##            plt.ylim([0.0, 1.05])
##            plt.xlabel('False Positive Rate')
##            plt.ylabel('True Positive Rate')
##            plt.title('Receiver operating characteristic to multi-class')
##            plt.legend(loc="lower right")
##            plt.savefig(str('ROC_LOW_VARIANCE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
##            plt.gcf().clear()
            
        df = pd.DataFrame({"classifer Name":classifier_list  ,"No. of feature selected":nfeature,'Feature selection Name':feature_list,'Accuracy':accuracy_list,'Precision':precision_list,'F1 score':f1_list,'recall':recall_list,'time':time_list})
        table["classifer Name"]=classifier_list    
        table["No. of feature selected"]=nfeature
        table['Feature selection Name']=feature_list
        table['Accuracy']=accuracy_list
        table['Precision']=precision_list
        table['F1 score']=f1_list
        table['recall']=recall_list
        table['time']=time_list
        
        plt.boxplot(np.array(accuracy_list))
        plt.savefig('Boxplot_accuracy_'+str(label)+'_'+str(data)[:-4])
        plt.gcf().clear()
        plt.boxplot(np.array(precision_list))
        plt.savefig('Boxplot_precision_'+str(label)+'_'+str(data)[:-4])
        plt.gcf().clear()
        plt.boxplot(np.array(f1_list))
        plt.savefig('Boxplot_f1_score_'+str(label)+'_'+str(data)[:-4])
        plt.gcf().clear()
        plt.boxplot(np.array(recall_list))
        plt.savefig('Boxplot_recall_'+str(label)+'_'+str(data)[:-4])
        plt.gcf().clear()

        plt.errorbar(features_list,reliefF_acc_list, 
             xerr=0.005,
             yerr=0.005,
             label='reliefF',
             fmt='-',
             color='g',
             ecolor='xkcd:salmon', elinewidth=1.5,
             capsize=5,
             #capthick=2
             )
        plt.xlabel("features")
        plt.ylabel("error")
        plt.title("errorbar")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(label+str(data)[:-4])
        print(table)
        df.to_excel(writer, sheet_name=sheet)
    writer.save()


