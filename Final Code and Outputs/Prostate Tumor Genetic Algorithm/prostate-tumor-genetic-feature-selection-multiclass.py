from __future__ import print_function
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import itertools
from itertools import cycle
from sklearn import cross_validation
from sklearn import linear_model
from scipy import stats
from itertools import compress
from genetic_selection import GeneticSelectionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
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
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIM
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
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

    


def main():

    datasets=[]
    xlsxfile=[]
    newdataname=[]
   
    for i in os.listdir("dataset"):
        datasets.append(i)
        xlsxfile.append(str(i)[:-4]+'.xlsx')
        newdataname.append('newdataset/'+str(i)[:-4]+'.csv')
    
    
    
    for data ,newdata in zip(datasets,newdataname):
      
##        df = pd.read_csv('dataset/'+data, header=None)
##        df.replace('?', np.NaN)
##        df = df.apply(pd.to_numeric, args=('coerce',))
##        df=df.apply(lambda x: x.fillna(x.mean()),axis=0)
##        X, Y = df.iloc[:, 1:].values, df.iloc[:, 0].values
        
        mat = scipy.io.loadmat('dataset/'+data)
        X = mat['data']
        Y = X[:, 0]
        print(Y)
        X1=X[:,1:]
##        print(Y)
            
        X1=np.array(X1)
        Y=np.array(Y)
        Y=Y.astype(int)
        X1=X1.astype(float)
        feature_numbers=np.linspace(1, len(X1[0]), len(X1[0]))
        nh=10
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
        #estimator =GenELMClassifier(hidden_layer=srhl_rbf)
        #MIM_sf,a,b=MIM.mim(X1,Y,n_selected_features=10)
        #mRMR_sf,a,b=MRMR.mrmr(X1,Y,n_selected_features=20)
        #Xtemp=X1[:,MIM_sf[0:(11)]]
        estimator=linear_model.LogisticRegression()
        selector = GeneticSelectionCV(estimator,
                                      cv=10,
                                      verbose=1,
                                      scoring="accuracy",
                                      n_population=30,
                                      crossover_proba=0.5,
                                      #0.5,
                                      mutation_proba=0.2,
                                      #0.2,
                                      n_generations=50,
                                      crossover_independent_proba=0.6,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      caching=True,
                                      n_jobs=-1)
        
        selector = selector.fit(X1, Y)
        y=selector.gen
        x1=selector.avg[:,0]
        x2=selector.mini[:,0]
        x3=selector.maxi[:,0]
        lw=1.5
        plt.plot(y,x1,label='Average',color='black' ,lw=lw)
        plt.plot(y,x2,label='Minimum',color='silver',lw=lw)
        plt.plot(y,x3,label='Maximum',color='gray',lw=lw)
        plt.ylabel("Statistics")
        plt.xlabel("Generations")
        plt.legend(loc="lower right")
        plt.savefig('accuracy_statistics_'+'_'+str(data)[:-4])
        plt.gcf().clear()

        features_in_gen=selector.avg[:,1]
        plt.plot(np.arange(0,51),features_in_gen,'ro')
        plt.ylabel("Features selected")
        plt.xlabel("Generations")
        plt.savefig('Number_of_Features_'+'_'+str(data)[:-4])
        plt.gcf().clear()

        writer = pd.ExcelWriter('accuracy_statistics_'+'_'+str(data)[:-4]+'.xlsx', engine='xlsxwriter')
        df1 = pd.DataFrame({'Maximum Accuracy':x3.tolist(),'Minimum Accuracy':x2.tolist(),'Average Accuracy':x1.tolist()})
        df1.to_excel(writer, sheet_name='sheet1')
        writer.save()    
        
        
        features=selector.support_
        print(features)
    
        feature_numbers=np.array(list(compress(feature_numbers, features)))
        feature_numbers=feature_numbers.astype(int)
        feature_numbers[:] = [x - 1 for x in feature_numbers]
        print(features.shape)
        print(feature_numbers)
        X_new=X1[:,feature_numbers[:]]
        X_new=X_new.tolist()
        Y=Y.tolist()
        print(Y)
        data_new=[]
        for (row,y) in zip(X_new,Y):
            row.insert(0,y)
            data_new.append(row)
        
        data_frame = pd.DataFrame(data_new)
        
        data_frame.to_csv(newdata, header=False, index=False)
        
    newdatasets=[]
    for i in os.listdir("newdataset"):
        newdatasets.append(i)
        

    model1 = SVC(kernel='poly',probability=True) 
    model2 = RandomForestClassifier()
    model3 = DecisionTreeClassifier(random_state=0)
    model4 = BaggingClassifier(random_state=0)
    ##model5 = GaussianNB()
    ##model6 = GradientBoostingClassifier(random_state=0)
    #model7 = KNeighborsClassifier()

    for data ,outfile in zip(newdatasets,xlsxfile):
        color=['black','red','blue','orange','brown','cyan','yellow','violet','pink','green','gold','gray','magenta','navy','yellowgreen','limegreen']
        #getting the data
##      to read from csv
        df = pd.read_csv('newdataset/'+data, header=None)
        X, Y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    
           
        X=np.array(X)
        Y=np.array(Y)
        Y=Y.astype(int)
        X=X.astype(float)

        #normalization
        #X=stats.zscore(X)
        
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
        names=['SVM','Random Forest','Desicion Tree','Bagging']
        svm_error_list=[]
        RF_error_list=[]
        DT_error_list=[]
        bagging_error_list=[]
        
        for model, label,sheet ,elist in zip([ model1,model2,model3,model4],names,['sheet1','sheet2','sheet3','sheet4'],[svm_error_list,RF_error_list,DT_error_list,bagging_error_list]):
            
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
            classifier1 = OneVsRestClassifier(model)
            
            table=pd.DataFrame()
            ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
            skf=StratifiedKFold(n_splits=10,shuffle=True)
    ##        loo = LeaveOneOut()
            
            
            y_train=[]
            y_test=[]
            X_train=[]
            X_test=[]
            prediction=[]
            
            for train, test in skf.split(X,Y):
                y_train=Y[train]
                y_test=Y[test]
                X_train=X[train]
                X_test=X[test]
                B=model.fit(X[train], Y[train])
                prediction=B.predict(X[test])
                acc = accuracy_score(Y[test], prediction)
                #m=matthews_corrcoef(Y[test], prediction)
                pre=precision_score(Y[test], prediction)
                pre=pre
                rec=recall_score(Y[test], prediction)
                rec=rec
                f=f1_score( Y[test], prediction)
                f=f
                
                accuracy_list.append(acc)
                elist.append(1-acc)
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
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1])
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
            df.to_excel(writer, sheet_name=sheet)
            print(table)
        writer.save()
        folds=[1,2,3,4,5,6,7,8,9,10]
        for elist ,name,color in zip([svm_error_list,RF_error_list,DT_error_list,bagging_error_list],names,['brown','cyan','red','violet']):
            print(elist)
            print(folds)
            plt.errorbar(folds,elist, 
                 xerr=0.005,
                 yerr=0.005,
                 label=name,
                 fmt='-',
                 color=color,
                 ecolor='black', elinewidth=1.5,
                 capsize=5,
                 #capthick=2
                 )
            plt.xlabel("folds")
            plt.ylabel("error")
            plt.title("errorbar")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
        plt.grid()
        plt.savefig('error_bar_'+label+'_'+str(data)[:-4], bbox_inches='tight')
        plt.gcf().clear()
            
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
        
if __name__ == "__main__":
    main()
