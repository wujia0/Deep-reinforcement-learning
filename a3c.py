import tensorflow as tf
import numpy as np
from greedy_policy import GreedyPolicy
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 190
BATCH_SIZE = 19
GAMMA = 1


class DDPG:
    """docstring for DDPG"""
    def __init__(self):
        self.name = 'DDPG' # name for uploading results
        self.time_step=0
        self.state_dim = 375
        self.action_dim = 1
        self.n_outputs=2

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim,self.n_outputs)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = GreedyPolicy(self.action_dim,self.n_outputs)

    def train(self):

        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        discrete_action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch]) 

        # for action_dim = 1
        discrete_action_batch = np.resize(discrete_action_batch,[BATCH_SIZE,self.action_dim])
        # Calculate y_batch
        
        next_discrete_action_batch = self.actor_network.target_discrete_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_discrete_action_batch)
        y_batch = []  
        for i in range(len(minibatch)):  
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L

        self.critic_network.train(y_batch,state_batch,discrete_action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)    


        self.actor_network.train(q_gradient_batch,state_batch)
        if self.time_step%19==0:
            print('steps:',self.time_step)
            print('action_batch_for_gradients:',np.reshape(action_batch_for_gradients,(1,BATCH_SIZE)))

            self.critic_network.summary(y_batch,state_batch,discrete_action_batch)
            self.actor_network.summary(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_discrete_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        discrete_action = self.actor_network.discrete_action(state)
        return self.exploration_noise.generate(discrete_action,self.critic_network.time_step)

    def action(self,state):
        action = self.actor_network.action(state)
        return action
    
    def discrete_action(self,state):
        discrete_action = self.actor_network.discrete_action(state)
        return discrete_action
    
    def perceive(self,state,discrete_action,reward,next_state,done):

        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,discrete_action,reward,next_state,done)

        if self.replay_buffer.count() >=  REPLAY_START_SIZE:
            self.time_step+=1
            if self.time_step%19==0:
                self.train()

#        if self.time_step%120==1:        
#            self.critic_network.summary_q(np.resize(state,[1,375]),np.resize(discrete_action,[1,1]))

        if self.time_step % 600 == 0 and self.time_step > 600:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)












