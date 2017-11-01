import pandas as pd
import os
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
    n_samples, n_features = X.shape
            

    
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
        feature_list.append("reliefF")
        nfeature.append(i)
        classifier_list.append(classifier)
        
        classifier1 = OneVsRestClassifier(model)

        
        features_list=[10,20,40,60,80,100]
        for i in features_list:
            

            print("reliefF feature selection algorithm ")
            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
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
            
            for train, test in ss:
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
            reliefF_acc_list.append(1-accuracy)
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


            
        df = pd.DataFrame({"classifer Name":classifier_list  ,"No. of feature selected":nfeature,'Feature selection Name':feature_list,'Accuracy':accuracy_list,'Precision':precision_list,'F1 score':f1_list,'recall':recall_list,'time':time_list})
        table["classifer Name"]=classifier_list    
        table["No. of feature selected"]=nfeature
        table['Feature selection Name']=feature_list
        table['Accuracy']=accuracy_list
        table['Precision']=precision_list
        table['F1 score']=f1_list
        table['recall']=recall_list
        #table['matthews_corrcoef']=matthews_corrcoef_list
        table['time']=time_list

        #boxplots
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

        #error bars
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
        plt.savefig('error_bar'+label+'reliefF'+str(data)[:-4])
        plt.gcf().clear()

        
        print(table)
        df.to_excel(writer, sheet_name=sheet)
    writer.save()


