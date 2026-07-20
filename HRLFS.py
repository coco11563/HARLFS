import time
import torch
import argparse
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm
from utils.networks import ActorCritic
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from utils.preprocess import min_max_standardize,load_train_and_test,get_dataset_type,concat_LLM_embedding
from utils.evaluate import evaluate_by_randomforest,evaluate_features
warnings.filterwarnings('ignore')


"""
21 datasets
'spectf','svmguide3','german_credit','credit_default','spam_base',
'megawatt1','ionosphere','openml_586','openml_589',
'openml_607','openml_616','openml_618','openml_620','openml_637','mice_protein',
'coil-20','mnist','otto','jannis','cao','han'
"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="run method in this dataset", default="spam_base")


@dataclass
class Configuration:
    EXPLORE_STEPS:int=200 
    LEARN_STEPS:int=200 
    ALPHA=0.4 
    LAMBDA=0.6 
    N_ACTIONS:int=2 
    EPSILON:float = 0.9 
    N_ESTIMATOR:int=1 
    MEMORY_CAPACITY:int=400 
    RANDOM_STATE:int=42 
    TARGET_REPLACE_ITER:int=50 
    DEVICE:str='cpu' 
    SHOW=True 
    LOG=True

    def get(self,*propertys):
        return {p:self.__dict__[p] for p in propertys}



class Agent:
    id:int
    action:int
    policy:ActorCritic
    config:Configuration
    is_leaf=True
    n_children:int=0
    left,right=None,None

    def __init__(self,id,N_STATES,config:Configuration) -> None:
        self.id=id
        self.config=config
        self.policy=ActorCritic(N_STATES=N_STATES,**config.get('N_ACTIONS','MEMORY_CAPACITY','TARGET_REPLACE_ITER','DEVICE'))



class MAHRLFS:
    X,y=None,None
    n_feature:int=0
    nodes,agents=[],[]
    task_type:str
    dataset:str
    config:Configuration


    def __init__(self,X:np.ndarray,y:np.ndarray,dataset,task_type,config:Configuration) -> None:
        self.X=X
        self.y=y
        self.task_type=task_type
        self.dataset=dataset
        self.n_feature=X.shape[1]
        self.config=config


    def select_feature(self,agent:Agent,state,action):
        scale=agent.n_children/10
        threshold=1/(1+np.exp(-10*(scale-0.5)))/2+0.5
    
        n_active=1 if action==1 and (not agent.is_leaf) else 0
        action=agent.policy.choose_action(state,threshold=threshold,epsilon=self.config.EPSILON) if action==1 else 0
        agent.action=action

        if not agent.is_leaf:
            n1=self.select_feature(agent.left,state,action)
            n2=self.select_feature(agent.right,state,action)
            return n1+n2+n_active
        else:
            return n_active

    
    def data2state_by_hybrid_feature(self,actions,n_components,random_state=42):
    
        X=self.X.copy()
        X[:,actions==0]=0
        gmm = GaussianMixture(n_components=n_components, random_state=random_state,covariance_type='diag')
        gmm.fit(X)

        means = gmm.means_  
        covariances = gmm.covariances_  
        weights = gmm.weights_.reshape(-1,1)  

        means_=np.sum(means*weights,axis=0)
        covariances_=np.sum(covariances*weights,axis=0)
        means_=min_max_standardize(means_).reshape(1,-1)
        covariances_=min_max_standardize(covariances_).reshape(1,-1)
        state=concat_LLM_embedding(self.dataset,means_,covariances_,actions=actions).flatten()
        
        state=torch.Tensor(state).to(torch.float32).to(self.config.DEVICE)

        return state
    
    
    def report(self,record):
        content={
            'dataset':record['dataset'],
            'stage':record['stage'],
            'reward_current':record['reward'],
        }
        if self.config.SHOW:
            self.range_bar.set_postfix(**content)
    

        return
    

    def run(self,):
        start=time.time()
        n_feature=self.n_feature
        n_sample=self.X.shape[0]
        config=self.config

        # Hybrid Feature State Extraction
        n_components=len(np.unique(self.y)) if self.task_type!='r' else 10 
        gmm = GaussianMixture(n_components=n_components, random_state=config.RANDOM_STATE,covariance_type='diag')
        gmm.fit(self.X)
        means = gmm.means_  
        covariances = gmm.covariances_  
        data=concat_LLM_embedding(self.dataset,means,covariances)

        # Divide-and-Conquer Agent Architecture
        agg_cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        agg_cluster_model.fit(data)
        children = agg_cluster_model.children_
        agents=[Agent(i,self.X.shape[1]*2,self.config) for i in range(2*n_feature-1)]
        for i in range(n_feature):
            agents[i].n_children=1
        for idx,pair in enumerate(children):
            agents[idx+n_feature].left=agents[pair[0]]
            agents[idx+n_feature].right=agents[pair[1]]
            agents[idx+n_feature].is_leaf=False
            agents[idx+n_feature].n_children=agents[pair[0]].n_children+agents[pair[1]].n_children 
        actions=np.random.randint(0,2,size=(n_feature,))
        state=self.data2state_by_hybrid_feature(actions,n_components,self.config.RANDOM_STATE)
        self.range_bar=range(config.EXPLORE_STEPS+config.LEARN_STEPS)
        if self.config.SHOW:
            self.range_bar=tqdm(range(config.EXPLORE_STEPS+config.LEARN_STEPS)) 
        metric_optimal=0
        records=[] 
        info={}
        for key in self.config.__dict__:
            if key=='logger':
                continue
            info[key]=self.config.__dict__[key]
       
        # Exploration and Optmization
        for step in self.range_bar:
            n_active=self.select_feature(agents[-1],state,1)
            actions=np.array([agent.action for agent in agents[:n_feature]])
            if actions.sum()==0:
                actions=np.random.randint(0,2,size=(n_feature,))
            state_next=self.data2state_by_hybrid_feature(actions,n_components,self.config.RANDOM_STATE)
            feature_selected=np.where(actions==1)[0]
            X_=self.X[:,feature_selected]
            r=evaluate_by_randomforest(X_,self.y,self.task_type,n_estimator=self.config.N_ESTIMATOR,random_state=config.RANDOM_STATE)
            reward=self.reward(r,len(feature_selected))
            if reward>metric_optimal:
                metric_optimal=reward
                n_feature_optimal=len(feature_selected)
            for agent in agents:
                agent.policy.store_transition(state,agent.action,r,state_next)
            state=state_next
            stage="exploration" if step<self.config.EXPLORE_STEPS else "optimization"
            records.append(record:={
                'n_active':n_active,
                'step':step,
                'dataset':self.dataset,
                'dataset_type':self.task_type,
                'feature_selected':feature_selected,
                'reward':r,
                'time':time.time()-start,
                'n_feature_selected':len(feature_selected),
                'n_feature':n_feature,
                'n_sample':n_sample,
                'stage': stage,
                **info
            })
        
            self.report(record)
            if step <self.config.EXPLORE_STEPS:
                continue
            for agent in agents:
                agent.policy.learn()

        records=sorted(records,key=lambda x:x['reward'],reverse=True)

        return records
    
    
    def reward(self,performance,n_feature_selected):
        reward = (
            self.config.ALPHA*performance + 
            (1-self.config.ALPHA)*(self.n_feature - n_feature_selected) / (self.n_feature + self.config.LAMBDA*n_feature_selected)
        ) 

        return reward



if __name__ =='__main__':
    
    config=Configuration()
    args = parser.parse_args()
    dataset=args.dataset
    X_train,y_train,X_test,y_test=load_train_and_test(dataset)
    datset_type=get_dataset_type(dataset)
    model=MAHRLFS(X_train,y_train,dataset,datset_type,config)
    records=model.run()
    records=evaluate_features(records,X_test,y_test,10,datset_type,config.RANDOM_STATE,config.SHOW)
    print(f"the metric of optimal feature subset is {records[0]['metric']}")

 


