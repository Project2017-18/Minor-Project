import pandas as pd
import os
from itertools import cycle
from sklearn import cross_validation
from scipy import stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import model_selection
#classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
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
model3 = DecisionTreeClassifier(random_state=0)
model4 = BaggingClassifier(random_state=0)
##model5 = GaussianNB()
##model6 = GradientBoostingClassifier(random_state=0)
##model7 = KNeighborsClassifier()

for data ,outfile in zip(datasets,xlsxfile):
    color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
    #getting the data
    mat = scipy.io.loadmat('dataset/'+data)
    X = mat['data']
    Y = X[:, 0]
    Y=Y.astype(int)
    X=X[:,1:]
    a=np.unique(Y)
    Y1 = label_binarize(Y, classes=a.tolist())
    n_classes = a.size
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    colors = cycle(color[0:n_classes])
    n_samples, n_features = X.shape
    
    accuracy_box_plot=[]
    precision_box_plot=[]
    recall_box_plot=[]
    f1score_box_plot=[]
    names=['svm','randomforest','Decision tree','Bagging classifier']
    for model, label,sheet in zip([ model1,model2,model3,model4],names,['sheet1','sheet2','sheet3','sheet4']):
        
        start=time.time()
        accuracy_list=[]
        precision_list=[]
        recall_list=[]
        f1_list=[]
        feature_list=[]
        nfeature=[]
        classifier_list=[]
        time_list=[]
        #matthews_corrcoef_list=[]
                
        CMIM_ERROR_LIST=[]
        MRMR_ERROR_LIST=[]
        ICAP_ERROR_LIST=[]
        DISR_ERROR_LIST=[]
        MIM_ERROR_LIST=[]
        MIFS_ERROR_LIST=[]
        JMI_ERROR_LIST=[]
        CIFE_ERROR_LIST=[]
        LCSI_ERROR_LIST=[]
        
        fisher_score_ERROR_LIST=[]
        reliefF_ERROR_LIST=[]
        SPEC_ERROR_LIST=[]
        TRACE_RATIO_ERROR_LIST=[]
        LAP_SCORE_ERROR_LIST=[]

        CFS_ERROR_LIST=[]
        CHI_SQUARE_ERROR_LIST=[]
        F_SCORE_ERROR_LIST=[]
        GINI_INDEX_ERROR_LIST=[]
        LOW_VARIANCE_ERROR_LIST=[]
        
        classifier=label
        table=pd.DataFrame()
        ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
        
        
        accuracy=0
        precision=0
        recall=0
        f1score=0
        #mcc=0
        
        for train, test in ss:
            B=model.fit(X[train], Y[train])
            prediction=B.predict(X[test])
            acc = accuracy_score(Y[test], prediction)
            accuracy=accuracy+acc
            #m=matthews_corrcoef(Y[test], prediction)
            #mcc=mcc+m
            pre=precision_score(Y[test], prediction,average=None)
            precision=precision+pre.mean()
            rec=recall_score(Y[test], prediction,average=None)
            recall=recall+rec.mean()
            f=f1_score( Y[test], prediction,average=None)
            f1score=f1score+f.mean()
            
        
        accuracy=float(accuracy)/10
        precision=float(precision)/10
        recall=float(recall)/10
        f1score=float(f1score)/10
        #mcc=float(mcc)/10
        accuracy_list.append(accuracy)
        #matthews_corrcoef_list.append(mcc)
        precision_list.append( precision)
        recall_list.append(recall)
        f1_list.append(f1score)
        time_list.append(float(time.time()-start))
        start=time.time()
        feature_list.append("Without feature selection")
        nfeature.append(n_features)
        classifier_list.append(classifier)
        
        classifier1 = OneVsRestClassifier(model)

        
        features_list=[10,20,40,60,80,100]
        for i in features_list:
            
##              #information_theoretical_based
##            print("CMIM feature selection algorithm ")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                CMIM_sf,a,b=CMIM.cmim(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,CMIM_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            CMIM_ERROR_LIST.append(accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("CMIM")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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

##            print("MRMR feature selection algorithm ")
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                mRMR_sf,a,b=MRMR.mrmr(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,mRMR_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            MRMR_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("MRMR")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##
##            
##            print("ICAP feature selection algorithm")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                ICAP_sf,a,b=ICAP.icap(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,ICAP_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            ICAP_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("ICAP")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            skf=StratifiedKFold(n_splits=2,shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in skf.split(X,Y):
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                DISR_sf,a,b=DISR.disr(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,DISR_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            DISR_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("DISR")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                
##                MIM_sf,a,b=MIM.mim(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,MIM_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            MIM_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("MIM")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                MIFS_sf,a,b=MIFS.mifs(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,MIFS_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            MIFS_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("MIFS")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                JMI_sf,a,b=JMI.jmi(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,JMI_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            JMI_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("JMI")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                CIFE_sf,a,b=CIFE.cife(X[train],Y[train],n_selected_features=i)
##                Xtemp=X[:,CIFE_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            CIFE_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("CIFE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##           
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                kwargs = {'function_name': 'JMI','n_selected_features':i}
##                LCSI_sf,a,b=LCSI.lcsi(X[train],Y[train],**kwargs)
##                Xtemp=X[:,LCSI_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            LCSI_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("LCSI")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            #similarity_based
            print("FISHER_SCORE feature selection algorithm ")
            

            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            skf = StratifiedKFold(n_splits=10,shuffle=True)
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
            
            for train, test in skf.split(X,Y):
                y_train=Y[train]
                y_test=Y[test]
                X_train=X[train]
                X_test=X[test]
                score1 = fisher_score.fisher_score(X[train], Y[train])
                idx = fisher_score.feature_ranking(score1)
                Xtemp=X[:,idx[0:(i+1)]]
                B=model.fit(Xtemp[train], Y[train])
                prediction=B.predict(Xtemp[test])
                acc = accuracy_score(Y[test], prediction)
                accuracy=accuracy+acc
                #m=matthews_corrcoef(Y[test], prediction)
                #mcc=mcc+m
                pre=precision_score(Y[test], prediction,average=None)
                precision=precision+pre.mean()
                rec=recall_score(Y[test], prediction,average=None)
                recall=recall+rec.mean()
                f=f1_score( Y[test], prediction,average=None)
                f1score=f1score+f.mean()
                
            
            accuracy=float(accuracy)/10
            precision=float(precision)/10
            recall=float(recall)/10
            f1score=float(f1score)/10
            #mcc=float(mcc)/10
            accuracy_list.append(accuracy)
            #matthews_corrcoef_list.append(mcc)
            fisher_score_ERROR_LIST.append(1-accuracy)
            precision_list.append( precision)
            recall_list.append(recall)
            f1_list.append(f1score)
            time_list.append(float(time.time()-start))
            start=time.time()
            feature_list.append("FISHER_SCORE")
            nfeature.append(i)
            classifier_list.append(classifier)
            
            y_train = label_binarize(y_train, classes=a.tolist())
            y_test = label_binarize(y_test, classes=a.tolist())
            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            plt.savefig(str('ROC_FISHER_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
            plt.gcf().clear()

            print("reliefF feature selection algorithm ")

            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            skf = StratifiedKFold(n_splits=10,shuffle=True)
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
            
            for train, test in skf.split(X,Y):
                y_train=Y[train]
                y_test=Y[test]
                X_train=X[train]
                X_test=X[test]
                score1 = reliefF.reliefF(X[train], Y[train])
                idx = reliefF.feature_ranking(score1)
                Xtemp=X[:,idx[0:(i+1)]]
                B=model.fit(Xtemp[train], Y[train])
                prediction=B.predict(Xtemp[test])
                acc = accuracy_score(Y[test], prediction)
                accuracy=accuracy+acc
                #m=matthews_corrcoef(Y[test], prediction)
                #mcc=mcc+m
                pre=precision_score(Y[test], prediction,average=None)
                precision=precision+pre.mean()
                rec=recall_score(Y[test], prediction,average=None)
                recall=recall+rec.mean()
                f=f1_score( Y[test], prediction,average=None)
                f1score=f1score+f.mean()
                
            
            accuracy=float(accuracy)/10
            precision=float(precision)/10
            recall=float(recall)/10
            f1score=float(f1score)/10
            #mcc=float(mcc)/10
            accuracy_list.append(accuracy)
            #matthews_corrcoef_list.append(mcc)
            reliefF_ERROR_LIST.append(1-accuracy)
            precision_list.append( precision)
            recall_list.append(recall)
            f1_list.append(f1score)
            time_list.append(float(time.time()-start))
            start=time.time()
            feature_list.append("reliefF")
            nfeature.append(i)
            classifier_list.append(classifier)
            
            y_train = label_binarize(y_train, classes=a.tolist())
            y_test = label_binarize(y_test, classes=a.tolist())
            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                kwargs = {'style': 0}
##                score1 = SPEC.spec(X[train], **kwargs)
##                idx = SPEC.feature_ranking(score1, **kwargs)
##                Xtemp=X[:,idx[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            SPEC_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("SPEC")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            skf = StratifiedKFold(n_splits=10,shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in skf.split(X,Y):
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                idx, feature_score, subset_score = trace_ratio.trace_ratio(X[train], Y[train], i, style='fisher')
##                Xtemp=X[:,idx[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            TRACE_RATIO_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("TRACE_RATIO")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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

##            print("LAP_SCORE feature selection algorithm ")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            skf = StratifiedKFold(n_splits=2,shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in skf.split(X,Y):
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
##                W = construct_W.construct_W(X[train], **kwargs_W)
##                score1 = lap_score.lap_score(X[train], W=W)
##                idx = lap_score.feature_ranking(score1)
##                Xtemp=X[:,idx[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            LAP_SCORE_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("LAP_SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
##            
##            #statistical_based
##              #takes more time to run
##            print("CFS feature selection algorithm ")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                CFS_sf=CFS.cfs(X[train],Y[train])
##                Xtemp=X[:,CFS_sf[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            CFS_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("CFS")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            
##            print("CHI_SQUARE feature selection algorithm ")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            skf = StratifiedKFold(n_splits=10,shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in skf.split(X,Y):
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                score1=chi_square.chi_square(X,Y)
##                #score1=stats.zscore(X,axis=None,ddof=0)
##                idx = chi_square.feature_ranking(score1)
##                Xtemp=X[:,idx[0:i]]
##                
##                #Xtemp[train] = Xtemp[train].reshape((n_samples, n_features) )
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            CHI_SQUARE_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("CHI_SCORE")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            print("F_SCORE feature selection algorithm ")
            

            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            
            skf = StratifiedKFold(n_splits=10,shuffle=True)
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
            
            for train, test in skf.split(X,Y):
                y_train=Y[train]
                y_test=Y[test]
                X_train=X[train]
                X_test=X[test]
                score1 = f_score.f_score(X[train], Y[train])
                idx = f_score.feature_ranking(score1)
                Xtemp=X[:,idx[0:(i+1)]]
                B=model.fit(Xtemp[train], Y[train])
                prediction=B.predict(Xtemp[test])
                acc = accuracy_score(Y[test], prediction)
                accuracy=accuracy+acc
                #m=matthews_corrcoef(Y[test], prediction)
                #mcc=mcc+m
                pre=precision_score(Y[test], prediction,average=None)
                precision=precision+pre.mean()
                rec=recall_score(Y[test], prediction,average=None)
                recall=recall+rec.mean()
                f=f1_score( Y[test], prediction,average=None)
                f1score=f1score+f.mean()
                
            
            accuracy=float(accuracy)/10
            precision=float(precision)/10
            recall=float(recall)/10
            f1score=float(f1score)/10
            #mcc=float(mcc)/10
            accuracy_list.append(accuracy)
            #matthews_corrcoef_list.append(mcc)
            F_SCORE_ERROR_LIST.append(1-accuracy)
            precision_list.append( precision)
            recall_list.append(recall)
            f1_list.append(f1score)
            time_list.append(float(time.time()-start))
            start=time.time()
            feature_list.append("F_SCORE")
            nfeature.append(i)
            classifier_list.append(classifier)
            
            y_train = label_binarize(y_train, classes=a.tolist())
            y_test = label_binarize(y_test, classes=a.tolist())
            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            plt.savefig(str('ROC_F_SCORE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
            plt.gcf().clear()
##
##            print("GINI_INDEX feature selection algorithm ")
##            
##
##            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
##            #loo = LeaveOneOut()
##            accuracy=0
##            precision=0
##            recall=0
##            f1score=0
##            #mcc=0
##            y_train=[]
##            y_test=[]
##            X_train=[]
##            X_test=[]
##            
##            for train, test in ss:
##                y_train=Y[train]
##                y_test=Y[test]
##                X_train=X[train]
##                X_test=X[test]
##                score1 = gini_index.gini_index(X[train],Y[train])
##                idx = gini_index.feature_ranking(score1)
##                Xtemp=X[:,idx[0:(i+1)]]
##                B=model.fit(Xtemp[train], Y[train])
##                prediction=B.predict(Xtemp[test])
##                acc = accuracy_score(Y[test], prediction)
##                accuracy=accuracy+acc
##                #m=matthews_corrcoef(Y[test], prediction)
##                #mcc=mcc+m
##                pre=precision_score(Y[test], prediction,average=None)
##                precision=precision+pre.mean()
##                rec=recall_score(Y[test], prediction,average=None)
##                recall=recall+rec.mean()
##                f=f1_score( Y[test], prediction,average=None)
##                f1score=f1score+f.mean()
##                
##            
##            accuracy=float(accuracy)/10
##            precision=float(precision)/10
##            recall=float(recall)/10
##            f1score=float(f1score)/10
##            #mcc=float(mcc)/10
##            accuracy_list.append(accuracy)
##            #matthews_corrcoef_list.append(mcc)
##            GINI_INDEX_ERROR_LIST.append(1-accuracy)
##            precision_list.append( precision)
##            recall_list.append(recall)
##            f1_list.append(f1score)
##            time_list.append(float(time.time()-start))
##            start=time.time()
##            feature_list.append("GINI_INDEX")
##            nfeature.append(i)
##            classifier_list.append(classifier)
##            
##            y_train = label_binarize(y_train, classes=a.tolist())
##            y_test = label_binarize(y_test, classes=a.tolist())
##            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            print("LOW_VARIANCE feature selection algorithm ")


            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            skf = StratifiedKFold(n_splits=10,shuffle=True)
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
            
            for train, test in skf.split(X,Y):
                y_train=Y[train]
                y_test=Y[test]
                X_train=X[train]
                X_test=X[test]
                p = 0.1 
                selected_features = low_variance.low_variance_feature_selection(X, p*(1-p))
                Xtemp=selected_features
                B=model.fit(Xtemp[train], Y[train])
                prediction=B.predict(Xtemp[test])
                acc = accuracy_score(Y[test], prediction)
                accuracy=accuracy+acc
                #m=matthews_corrcoef(Y[test], prediction)
                #mcc=mcc+m
                pre=precision_score(Y[test], prediction,average=None)
                precision=precision+pre.mean()
                rec=recall_score(Y[test], prediction,average=None)
                recall=recall+rec.mean()
                f=f1_score( Y[test], prediction,average=None)
                f1score=f1score+f.mean()
                
            
            accuracy=float(accuracy)/10
            precision=float(precision)/10
            recall=float(recall)/10
            f1score=float(f1score)/10
            #mcc=float(mcc)/10
            accuracy_list.append(accuracy)
            #matthews_corrcoef_list.append(mcc)
            LOW_VARIANCE_ERROR_LIST.append(1-accuracy)
            precision_list.append( precision)
            recall_list.append(recall)
            f1_list.append(f1score)
            time_list.append(float(time.time()-start))
            start=time.time()
            feature_list.append("LOW_VARIANCE")
            nfeature.append(i)
            classifier_list.append(classifier)
            
            y_train = label_binarize(y_train, classes=a.tolist())
            y_test = label_binarize(y_test, classes=a.tolist())
            #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
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
            plt.savefig(str('ROC_LOW_VARIANCE_'+str(label)+'_'+str(data)[:-4]+'_'+str(i)))
            plt.gcf().clear()
            
        df = pd.DataFrame({"classifer Name":classifier_list  ,"No. of feature selected":nfeature,'Feature selection Name':feature_list,'Accuracy':accuracy_list,'Precision':precision_list,'F1 score':f1_list,'recall':recall_list,'time':time_list})
        table["classifer Name"]=classifier_list    
        table["No. of feature selected"]=nfeature
        table['Feature selection Name']=feature_list
        table['Accuracy']=accuracy_list
        table['Precision']=precision_list
        table['F1 score']=f1_list
        table['recall']=recall_list
        table['time']=time_list
        
        accuracy_box_plot.append(accuracy_list)
        precision_box_plot.append(precision_list)
        recall_box_plot.append(recall_list)
        f1score_box_plot.append(f1_list)
           
        
        #[CMIM_ERROR_LIST,MRMR_ERROR_LIST,ICAP_ERROR_LIST,DISR_ERROR_LIST,MIM_ERROR_LIST,MIFS_ERROR_LIST,JMI_ERROR_LIST,CIFE_ERROR_LIST,LCSI_ERROR_LIST]
        #[fisher_score_ERROR_LIST,reliefF_ERROR_LIST,SPEC_ERROR_LIST,TRACE_RATIO_ERROR_LIST,LAP_SCORE_ERROR_LIST]    
        #[CFS_ERROR_LIST,CHI_SQUARE_ERROR_LIST,F_SCORE_ERROR_LIST,GINI_INDEX_ERROR_LIST,LOW_VARIANCE_ERROR_LIST]
        #to run CFS ADD CFS_ERROR_LIST IN FOR LOOP
        for elist ,name,color in zip([fisher_score_ERROR_LIST,reliefF_ERROR_LIST,LOW_VARIANCE_ERROR_LIST,F_SCORE_ERROR_LIST],feature_list[1:],['brown','cyan','red','violet']):
            plt.errorbar(features_list,elist, 
                 xerr=0.005,
                 yerr=0.005,
                 label=name,
                 fmt='-',
                 color=color,
                 ecolor='black', elinewidth=1.5,
                 capsize=5,
                 #capthick=2
                 )
            plt.xlabel("features")
            plt.ylabel("error")
            plt.title("errorbar")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
        plt.grid()
        plt.savefig('error_bar_'+label+'_'+str(data)[:-4], bbox_inches='tight')
        plt.gcf().clear()

        
        print(table)
        df.to_excel(writer, sheet_name=sheet)
    writer.save()
    plt.boxplot(accuracy_box_plot,labels=names,showfliers=False)
    plt.savefig('Boxplot_accuracy_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(precision_box_plot,labels=names,showfliers=False)
    plt.savefig('Boxplot_precision_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(recall_box_plot,labels=names,showfliers=False)
    plt.savefig('Boxplot_recall_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(f1score_box_plot,labels=names,showfliers=False)
    plt.savefig('Boxplot_fscore_'+str(data)[:-4])
    plt.gcf().clear()


