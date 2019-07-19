import generate_env
import dpg
import gc #garbage collection
import numpy as np
gc.enable()

EPISODES = 1000000
TEST = 50

def main():
    env = generate_env.generateEnv()
    agent = dpg.DPG()

    for episode in range(EPISODES):
        pressure,production = env.reset()
        done=False

        # Train
        for step in range(3,19):
            action = agent.noise_action(pressure,production)
#            action = agent.noise_action(pressure/4000,production/10000)
            next_pressure,next_production,reward = env.update(action,step)
            if step==18:
                done=True            
            agent.perceive(pressure,production,action,reward,next_pressure,next_production,done)
            pressure = next_pressure
            production=next_production

        # Testing
        if episode % 100 == 1:

                
            total_reward=0
            total_actions=[]
            pressure,production = env.reset()
            for j in range(3,19):

                discreted_action = agent.discrete_action(pressure,production)
                next_pressure,next_production,reward= env.update(discreted_action,j)
                pressure = next_pressure
                production=next_production                    
                total_reward += reward
                total_actions.append(discreted_action)
            with open('Test_Actions.txt','a') as f:                  
                f.write('\n'+str(episode)+' '+', '.join([str(i) for i in total_actions])+'\n')
                f.close 
            with open('Test_Total_Reward.txt','a') as f:
                f.write('\n'+str(episode)+' '+str(total_reward)+'\n')
                f.close             


if __name__ == '__main__':
    main()
