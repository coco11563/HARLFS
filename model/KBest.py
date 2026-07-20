import sys
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append("./")
sys.path.append("../")
from utils.preprocess import load_train_and_test,get_dataset_type
from utils.evaluate import evaluate_by_randomforest



class KBest:

    def __init__(self, X, y,**arg):  
        self.X = X
        self.y = y

    def run(self, n_selected=None):  
        if n_selected == None:
            n_selected = self.X.shape[1] // 20
        skb = SelectKBest(score_func=f_classif, k=int(n_selected))
        skb.fit(self.X, self.y)
        choice = skb.get_support()
        self.selected_features=np.argwhere(choice).flatten()
        self.scores=skb.scores_

        return self.selected_features
        
    def log(self, logger):
        pass


if __name__ == "__main__":
    task_name = "spam_base"
    X_train,y_train,X_test,y_test = load_train_and_test(task_name)
    kbest = KBest(X_train, y_train)
    index = kbest.run()
    print(kbest.scores[index],kbest.scores)
    X_=X_test[:,index]
    print(index,evaluate_by_randomforest(X_,y_test,get_dataset_type(task_name),100,random_state=42))
    
