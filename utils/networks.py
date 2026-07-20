import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
    

class ActorCritic(object):

    def __init__(self, N_STATES, N_ACTIONS,MEMORY_CAPACITY,TARGET_REPLACE_ITER,DEVICE,LR=0.01,GAMMA = 0.9,BATCH_SIZE=32):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.GAMMA=GAMMA
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.BATCH_SIZE=BATCH_SIZE
        self.device=DEVICE
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.memory = torch.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)).to(torch.float32).to(self.device)
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS).to(DEVICE).to(torch.float32), Net(N_STATES, N_ACTIONS).to(DEVICE).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()


    def choose_action(self, x:torch.Tensor,threshold,epsilon):
        
        if np.random.rand()<epsilon: 
            x = torch.unsqueeze(x, 0).to(torch.float32).to(self.device)
            logits = self.eval_net.forward(x)
            prob=torch.softmax(logits,1)
            return 0 if prob[0][0]>threshold else 1
        else:
            return 0 if np.random.rand()>threshold else 1



    def store_transition(self, s, a, r, s_):

        transition = torch.hstack((s, torch.Tensor([a, r],device=self.device), s_)).to(torch.float32).to(self.device)
        index = self.memory_counter % self.MEMORY_CAPACITY  
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :self.N_STATES].to(torch.float32)
        b_a = b_memory[:, self.N_STATES:self.N_STATES + 1].to(torch.long)
        b_r = b_memory[:, self.N_STATES + 1:self.N_STATES + 2].to(torch.float32)
        b_s_ = b_memory[:, -self.N_STATES:].to(torch.float32)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

