import tensorflow as tf
import numpy as np
from greedy_policy import GreedyPolicy
from critic_network import CriticNetwork 
from replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 1


class DPG:
    """docstring for DPG"""
    def __init__(self):
        self.name = 'DPG' # name for uploading results
        self.time_step=0
        self.state_dim = 375
        self.n_outputs=2

        self.sess = tf.InteractiveSession()

#        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim,self.n_outputs)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.n_outputs)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = GreedyPolicy(self.n_outputs)

    def train(self):

        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        pressure_batch = np.asarray([data[0] for data in minibatch])
#        pressure_batch=pressure_batch/4000
        
        production_batch = np.asarray([data[1] for data in minibatch])
#        production_batch=production_batch/10000
        
        action_batch = np.asarray([data[2] for data in minibatch])
        action_batch = np.resize(action_batch,[BATCH_SIZE,1])
                
        reward_batch = np.asarray([data[3] for data in minibatch])
#        reward_batch=reward_batch/10e6
        
        next_pressure_batch = np.asarray([data[4] for data in minibatch])
#        next_pressure_batch=next_pressure_batch/4000
        
        next_production_batch = np.asarray([data[5] for data in minibatch])
#        next_production_batch=next_production_batch/10000
        
        done_batch = np.asarray([data[6] for data in minibatch]) 

        # Calculate y_batch
        
        target_q_value_batch = self.critic_network.target_q(next_pressure_batch,next_production_batch)
#        print(target_q_value_batch,target_q_value_batch.shape)
        original_q_value_batch = self.critic_network.original_q_value(next_pressure_batch,next_production_batch)
#        print(original_q_value_batch,original_q_value_batch.shape)        
        max_action_batch = np.argmax(original_q_value_batch,axis=1)
        max_action_batch =  np.eye(self.n_outputs)[max_action_batch.reshape(-1)]            
        q_value_batch = np.sum(target_q_value_batch * max_action_batch, axis=1, keepdims=True)        
#        print(q_value_batch,q_value_batch.shape)        

        y_batch=reward_batch + (1.0-done_batch)*GAMMA * q_value_batch     


        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
#        print(y_batch,y_batch.shape)

        self.critic_network.train(y_batch,pressure_batch,production_batch,action_batch)
        
        if self.time_step%100==0:
            print('steps:',self.time_step)
            print('y:',np.reshape(y_batch,(1,BATCH_SIZE)))
            print('original_q_value:',original_q_value_batch[0:10])
            print('reward:',np.reshape(reward_batch,(1,BATCH_SIZE)))
            print('done:',np.reshape(done_batch,(1,BATCH_SIZE)))
            print('production:',production_batch[0:10])           
            self.critic_network.summary(y_batch,pressure_batch,production_batch,action_batch)


        # Update the target networks
        self.critic_network.update_target()        

#        self.critic_network.update_target()

    def noise_action(self,pressure,production):
        # Select action a_t according to the current policy and exploration noise

        q_value = self.critic_network.q_value(pressure,production)
        return self.exploration_noise.generate(q_value,self.time_step)

    def discrete_action(self,pressure,production):

        q_value = self.critic_network.q_value(pressure,production)
        return np.argmax(q_value,axis=1)
    
    def perceive(self,pressure,production,action,reward,next_pressure,next_production,done):

        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(pressure,production,action,reward,next_pressure,next_production,done)

        if self.replay_buffer.count() >=  REPLAY_START_SIZE:
            self.time_step+=1
            self.train()

        if self.time_step % 10000 == 0 and self.time_step >= 10000:
            self.critic_network.save_network(self.time_step)












