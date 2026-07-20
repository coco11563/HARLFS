import torch
import sys
from lassonet import LassoNetClassifierCV, LassoNetRegressorCV
from sklearn import preprocessing

sys.path.append("./")
sys.path.append("../")
from utils.preprocess import load_train_and_test,get_dataset_type
from utils.evaluate import evaluate_by_randomforest



def gen_lassonet(x, y,n_selected=None):
    if n_selected == None:
        n_selected = x.shape[1] // 5
    results = []
    normalizer = preprocessing.Normalizer()
    normalizer.fit(x)
    x = normalizer.transform(x)
    selector = LassoNetClassifierCV()  # LassoNetRegressorCV
    selector = selector.fit(x, y)
    scores = selector.feature_importances_
    value, indice = torch.topk(scores, n_selected)
    choice = torch.zeros(x.shape[1])
    choice[indice] = 1
   
    return choice


if __name__ == "__main__":
    task_name = "spam_base"
    X_train,y_train,X_test,y_test = load_train_and_test(task_name)
    index = gen_lassonet(X_train, y_train)
    print(index)
    X_=X_test[:,index]
    print(evaluate_by_randomforest(X_,y_test,get_dataset_type(task_name),100,random_state=42))


