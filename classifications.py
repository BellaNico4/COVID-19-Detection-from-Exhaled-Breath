from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV,GroupShuffleSplit

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.random import choice
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFECV,RFE
from sklearn.preprocessing import PolynomialFeatures,RobustScaler
from sklearn.tree import DecisionTreeClassifier
#import sklearn_relief as relief
#from skrebate import ReliefF
#from skrebate import SURF
import plotly.graph_objects as go
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold         


def train_test(clf_dict,X,y,clf_features,cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1),return_train_score = False,target='Covid',verbose=True):  
    accuracy_result=[]
    recall_result=[]
    precision_result=[]
    f1_result=[]
    i = 0
    list_df = []
    dict_score = {}
    for train_index,test_index in cv.split(X,y['Covid'],X['index']):
        
        X_train, X_test = X.iloc[train_index],X.iloc[test_index]
        y_train, y_test = y.iloc[train_index],y.iloc[test_index]

        X_train, X_test = X_train.reset_index(drop=True)[clf_features],X_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True)[target],y_test.reset_index(drop=True)

        i +=1
        y_no = y_train[y_train == 0]
        y_yes = y_train[y_train == 1].sample(n=len(y_no),replace=True)
        X_train = pd.concat([X_train.iloc[y_yes.index],X_train.iloc[y_no.index]])
        y_train = pd.concat([y_yes,y_no])

        X_test = X_test.groupby('index').mean()[clf_features]
        y_test = y_test.groupby('index').min()[target]
        
        #additional_features = [x for x in clf_features if x not in grouping_features]
        #X_test_grouping[additional_features] = X_test.groupby('index').min()[additional_features]
        #X_test = X_test_grouping        
        for label, clf in clf_dict.items():
            clf.fit(X_train.values, y_train.values)
            y_pred = clf.predict(X_test.values)
            y_proba = clf.predict_proba(X_test.values)[:,1]
            df_tested = pd.DataFrame()
            df_tested['Tested patients'] = y_test.index.values
            df_tested['Mispredicted'] = [1 if row[1] != pred else 0 for row,pred in zip(y_test.items(),y_pred)]
            df_tested['Run'] = i
            df_tested['Algorithm'] = label
            list_df.append(df_tested)
            acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            specificity = recall_score(y_test, y_pred, pos_label=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            if return_train_score:
                y_pred_tr = clf.predict(X_train.values)
                y_proba_tr = clf.predict_proba(X_train.values)[:,1]
                acc_tr = balanced_accuracy_score(y_train, y_pred_tr)
                prec_tr = precision_score(y_train, y_pred_tr)
                rec_tr = recall_score(y_train, y_pred_tr)
                f1_tr = f1_score(y_train, y_pred_tr)
                specificity_tr = recall_score(y_train, y_pred_tr, pos_label=0)
                roc_auc_tr = roc_auc_score(y_train, y_proba_tr)
            if verbose:
                print(f'Balanced Accuracy {label} (split {i}): {acc}')
                print(f'Precision {label} (split {i}): {prec}')
                print(f'Recall {label} (split {i}): {rec}')
                print(f'F1 score {label} (split {i}): {f1}')
                print(f'Specificity {label}: (split {i}) {specificity}')
                print(f'AUC-ROC {label} (split {i}): {roc_auc}\n\n')
            if label not in dict_score:
                dict_score[label] = {'acc_test' : [acc],
                                     'prec_test': [prec],
                                     'rec_test' : [rec],
                                     'f1_test':[f1],
                                     'spec_test':[specificity],
                                     'roc_auc_test':[roc_auc]
                                    }
                if return_train_score:
                    dict_score[label]['acc_train'] = []
                    dict_score[label]['prec_train'] = []
                    dict_score[label]['rec_train'] = []
                    dict_score[label]['f1_train'] = []
                    dict_score[label]['spec_train'] = []
                    dict_score[label]['roc_auc_train'] = []
            else:
                dict_score[label]['acc_test'].append(acc)
                dict_score[label]['prec_test'].append(prec)
                dict_score[label]['rec_test'].append(rec)
                dict_score[label]['f1_test'].append(f1)
                dict_score[label]['spec_test'].append(specificity)
                dict_score[label]['roc_auc_test'].append(roc_auc)
           
            if return_train_score:
                dict_score[label]['acc_train'].append(acc_tr)
                dict_score[label]['prec_train'].append(prec_tr)
                dict_score[label]['rec_train'].append(rec_tr)
                dict_score[label]['f1_train'].append(f1_tr)
                dict_score[label]['spec_train'].append(specificity_tr)
                dict_score[label]['roc_auc_train'].append(roc_auc_tr)
    return dict_score,list_df           

def print_scores(dict_score):
    for key,item in dict_score.items():
            print(f'Mean Results {key}')
            for measure_name, measure_list in item.items():
                    print(f'{measure_name}: {np.mean(measure_list)} +- {np.std(measure_list)}')
                    
            