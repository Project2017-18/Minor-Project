                                                ##Code for binary datasets in which class with label-1 is considered positive##
                                                                ##Remove id number attribute##

import pandas as pd
import os
from collections import defaultdict
import itertools
from itertools import cycle
from sklearn import cross_validation
from scipy import stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest',
               cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 #horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
datasets=[]
xlsxfile=[]
for i in os.listdir("dataset"):
    datasets.append(i)
    xlsxfile.append(str(i)[:-4]+'.xlsx')
##model1 = SVC(kernel='poly',probability=True) 
model2 = RandomForestClassifier()
##model3 = GaussianNB()
##model4 = KNeighborsClassifier()
model5 = DecisionTreeClassifier(random_state=0)
model6 = GradientBoostingClassifier()
model7 = BaggingClassifier()
model8 = AdaBoostClassifier(n_estimators=100)

for data ,outfile in zip(datasets,xlsxfile):
    color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
    #getting the data
##    to read from csv
    df = pd.read_csv('dataset/'+data, header=None)
    #df.replace('?', np.NaN)
    X, Y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X=np.array(X)
    Y=np.array(Y)
    Y=Y.astype(int)
    X=X.astype(float)
    

##    #to read from mat file
##    mat = scipy.io.loadmat('dataset/'+data)
##    X = mat['data']
##    Y = X[:, 0]
##    Y=Y.astype(int)
##    X=X[:,1:]

    #normalization
    X=stats.zscore(X)
    
    a=np.unique(Y)
    classnames=a.tolist()
    Y1 = label_binarize(Y, classes=classnames)
    n_classes = a.size
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    colors = cycle(color[0:n_classes])
    n_samples, n_features = X.shape

    d = defaultdict(int)
    for i in Y:
        d[i] += 1
    n = min(d.iteritems(), key=lambda x: x[1])
    
    accuracy_box_plot=[]
    precision_box_plot=[]
    recall_box_plot=[]
    f1score_box_plot=[]
    names=['Random Forest','Desicion Tree','Gradient Boosting','Bagging','AdaBoosting']
    for model, label,sheet in zip([ model2,model5,model6,model7,model8],names,['sheet1','sheet2','sheet3','sheet4','sheet5']):
        
        start=time.time()
        accuracy_list=[]
        precision_list=[]
        recall_list=[]
        f1_list=[]
        classifier_list=[]
        time_list=[]
        error_list=[]
        #matthews_corrcoef_list=[]
                
               
        classifier=label
        table=pd.DataFrame()
        ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
        skf=StratifiedKFold(n_splits=10,shuffle=True)
##        loo = LeaveOneOut()
        
        
        y_train=[]
        y_test=[]
        X_train=[]
        X_test=[]
        prediction=[]
        
        for train, test in ss:
            y_train=Y[train]
            y_test=Y[test]
            X_train=X[train]
            X_test=X[test]
            B=model.fit(X[train], Y[train])
            prediction=B.predict(X[test])
            acc = accuracy_score(Y[test], prediction)
            #m=matthews_corrcoef(Y[test], prediction)
            pre=precision_score(Y[test], prediction)
            rec=recall_score(Y[test], prediction)
            f=f1_score( Y[test], prediction)
            
            accuracy_list.append(acc)
            error_list.append(1-acc)
            #matthews_corrcoef_list.append(mcc)
            precision_list.append( pre)
            recall_list.append(rec)
            f1_list.append(f)
            time_list.append(float(time.time()-start))
            start=time.time()
            classifier_list.append(classifier)
        
        # Compute confusion matrix for last fold
        cnf_matrix = confusion_matrix(y_test, prediction)
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=classnames,title='Confusion matrix')
        plt.savefig(str('ConfusionMatrix_'+str(label)+'_'+str(data)[:-4]))
        plt.gcf().clear()

        
        y_score=model.predict_proba(X_test)
        #X_train, X_test, y_train, y_test = train_test_split(Xtemp,Y1,train_size=0.5,random_state=25)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:,0], pos_label=2)
        roc_auc = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(str('ROC'+'_'+str(label)+'_'+str(data)[:-4]))
        plt.gcf().clear()


            
            
            
        df = pd.DataFrame({"classifer Name":classifier_list  ,'Accuracy':accuracy_list,'Precision':precision_list,'F1 score':f1_list,'recall':recall_list,'time':time_list})
        table["classifer Name"]=classifier_list    
        table['Accuracy']=accuracy_list
        table['Precision']=precision_list
        table['F1 score']=f1_list
        table['recall']=recall_list
        table['time']=time_list
        
        accuracy_box_plot.append(accuracy_list)
        precision_box_plot.append(precision_list)
        recall_box_plot.append(recall_list)
        f1score_box_plot.append(f1_list)

        plt.errorbar([1,2,3,4,5,6,7,8,9,10],error_list, 
             xerr=0.005,
             yerr=0.005,
             label=label,
             fmt='-',
             color='g',
             ecolor='xkcd:salmon', elinewidth=1.5,
             capsize=5,
             #capthick=2
             )
        plt.xlabel("fold no")
        plt.ylabel("error")
        plt.title("errorbar")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig('error_bar_'+label+'_'+str(data)[:-4])
        plt.gcf().clear()
        
        print(table)
        df.to_excel(writer, sheet_name=sheet)
    writer.save()
    plt.boxplot(accuracy_box_plot,labels=names)
    plt.savefig('Boxplot_accuracy_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(precision_box_plot,labels=names)
    plt.savefig('Boxplot_precision_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(recall_box_plot,labels=names)
    plt.savefig('Boxplot_recall_'+str(data)[:-4])
    plt.gcf().clear()
    plt.boxplot(f1score_box_plot,labels=names)
    plt.savefig('Boxplot_fscore_'+str(data)[:-4])
    plt.gcf().clear()


