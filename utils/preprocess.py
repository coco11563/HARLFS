import os
import numpy as np
import pandas as pd


def load(dataset):
    df=pd.read_hdf(f"./data/{dataset}.hdf",key='raw_train')
    return df.iloc[:,:-1].to_numpy(),df.iloc[:,-1].to_numpy()


def concat_LLM_embedding(dataset,means,covariances,actions=None):
    embedding_path=f"./data/{dataset}_embedding.npy"
    description_path=f"./data/{dataset}_description.npy"
    if os.path.exists(embedding_path):
        embeddings=np.load(embedding_path).T
    else:
        from openai import OpenAI
        api_key=""
        client = OpenAI(api_key=api_key)
        descriptions=open(description_path,'r').read().split('\n')
        embeddings=[]
        for description in descriptions:
            completion = client.embeddings.create(
                model="text-embedding-v3",
                input=description,
                encoding_format="float"
            )
            embeddings.append(completion.model_dump()['data'][0]['embedding'])
        embeddings=np.array(embeddings)

    n=means.shape[1]
    if actions is not None:
        embeddings[:,[actions==0][:-1]]=0
    state=np.vstack([means,covariances,embeddings])[:-2,:].T
    return state


def load_train_and_test(dataset):
    path=f"./data/{dataset}.hdf"

    train=pd.read_hdf(path,key='raw_train')
    test=pd.read_hdf(path,key='raw_test')
    X_train,y_train=train.iloc[:,:-1].to_numpy(),train.iloc[:,-1].to_numpy()
    X_test,y_test=test.iloc[:,:-1].to_numpy(),test.iloc[:,-1].to_numpy()
 
    return X_train,y_train,X_test,y_test


def get_dataset_type(dataset):
   
    X,y=load(dataset)

    if 'openml' in dataset:
        return 'r'
    elif len(np.unique(y))==2:
        return 'c'
    else:
        return 'mc'


def min_max_standardize(data:np.ndarray):
    return  (data-np.min(data))/(np.max(data)-np.min(data))


