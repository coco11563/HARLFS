import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold



def OneSubstractRAE(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return 1-error


def weighted_f1(y_test,y_predict):
    return f1_score(y_test,y_predict,average='weighted')


def micro_f1(y_test,y_predict):
    return f1_score(y_test,y_predict,average='micro')


def evaluate_by_randomforest(X,y,task_type,n_estimator,random_state=42,n_cv=1):
    model=RandomForestRegressor if task_type=='r' else RandomForestClassifier
    rf_clf = model(n_estimators=n_estimator, random_state=random_state)

    criterion=metrics_function[task_type]

    if n_cv==1: 
        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=random_state,shuffle=True)

        rf_clf.fit(X_train, y_train) 

        y_pred = rf_clf.predict(X_test)

        metric=criterion(y_test, y_pred)
        return metric
    else:
        if task_type=='r':
            data_cv=KFold(n_splits=n_cv, shuffle=True, random_state=random_state).split(X)
        else:
            data_cv=StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state).split(X,y)

        metrics=[]
        for train_index, test_index in data_cv:
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]

            rf_clf.fit(X_train, y_train) 

            y_pred = rf_clf.predict(X_test)

            metric_=criterion(y_test, y_pred)
            metrics.append(metric_)

        metric=np.mean(metrics)
        return metric,metrics


metrics_function={
    'c':weighted_f1,
    'mc':micro_f1,
    'r':OneSubstractRAE # 1-RAE
}


def evaluate_features(records,X,y,k,task_type,random_state,show=True):

    for record in records:
        record['metric']=0
    metrics=[]
    bar=tqdm(records[:k]) if show else records[:k]
    for idx,record in enumerate(bar):
        feature=record['feature_selected']
        X_=X[:,feature]
        metric,metric_cv=evaluate_by_randomforest(X_,y,task_type,100,random_state,n_cv=5)
        if show:
            bar.set_postfix(metric_feature_current=metric,metric_feature_optimal=max(metrics) if idx!=0 else 0)
        records[idx]['metric']=metric
        metrics.append(metric)
        records[idx]['metrics']=metric_cv
    

    return sorted(records,key=lambda x:x['metric'],reverse=True)

