import numpy as np
import subprocess
import pandas as pd
import os
import time
from shutil import copyfile
import random

def calculate_reward(step):
    ro=70
    cwi=5
    cw=5
    cgi=1.5/10**6
    b=0.1
    production_data=pd.read_csv('mxspe009_Production.rwo',delim_whitespace=True,skiprows=6,header=None)
    production=production_data.values 
    reward=((production[step,1]-production[step-1,1])*ro-(production[step,2]-production[step-1,2])*cw\
            -(production[step,4]-production[step-1,4])*cwi-(production[step,5]-production[step-1,5])*cgi)\
             /(1+b)**(production[step,0]/360)
    next_production=np.array([(production[step,1]-production[step-1,1]),(production[step,2]-production[step-1,2]),1e-4*(production[step,3]-production[step-1,3])])
#    print(next_production.shape)
    return next_production, reward

def random_reservoir():
#    mean=1.3*np.ones(5)
#    variance=0.6
#    L=600
#    Nx=5
#    Ny=5
#
#    arr=np.zeros([Nx,Ny])
#    for i in range(Nx):
#        for j in range(Ny):
#            arr[i,j]=300*(np.abs(i-j))
#
#    co_var=variance*np.exp(-1/L*arr)
#    L=np.linalg.cholesky(co_var)
#
#    for i in range(75):
#        np.random.seed=42
#        Z=np.random.normal([0,0,0,0,0],[1,1,1,1,1],5)
#        arr=mean+np.matmul(L,Z)
#        arr=arr.reshape([1,5])
#        if i==0:
#            log_perm=arr
#        else:
#            log_perm=np.concatenate((log_perm,arr),axis=0)
#    
#    perm=10**log_perm
#    perm=pd.DataFrame(data=perm)
#    perm.to_csv('PERM.txt',sep=' ',index=False,header=['*PERMI','*ALL','','',''],float_format='%.1f')
    
    i=random.randint(0, 49)
    copyfile('PERM_'+str(i+1)+'.txt','PERM.txt')

class generateEnv():
    def __init__(self):
        pass

    def update(self,action,step):
        
        pressures=np.zeros((50,375))
        productions=np.zeros((50,3))        
        rewards=np.zeros((50,1))
        flag=True        
        for i in range(50):

            if  flag==True:
                with open('SCHEDUEL.txt','r') as f:
                    file_data=f.readlines()
                    file_data[2*step-1]='*TIME '+str((step-1)*90)+'\n'
                    if action==1:
                        file_data[2*step]="OPEN 'Injector'\n"
                    if action==0:
                        file_data[2*step]="SHUTIN 'Injector'\n"
                    file_data[2*step+1],file_data[2*step+2],file_data[2*step+3]='WRST TIME\n','WPRN WELL TIME\n','WSRF GRID TIME\n'
                    file_data.append('*TIME '+str(step*90)+'\n')
                    file_data.append('*STOP\n')           
                    f.close
                with open('SCHEDUEL.txt','w') as f:
                    f.writelines(file_data)
                    f.close
                flag=False
            with open('mxspe009_Pressure.rwd','w') as f:
                file_data=["*FILES 'mxspe009_"+str(step)+"_perm_"+str(i+1)+".irf'\n", '\n', "*PROPERTY-FOR 'Pressure' " + str(step*90) + " *SRF-FORMAT\n"]
                f.writelines(file_data)
                f.close
            with open('mxspe009_Production.rwd','r') as f:
                file_data=f.readlines()
                file_data[0]="*FILES 'mxspe009_"+str(step)+"_perm_"+str(i+1)+".irf'\n"
                f.close
            with open('mxspe009_Production.rwd','w') as f:
                f.writelines(file_data)
                f.close
            
            subprocess.run(['C:\\Program Files (x86)\\CMG\\IMEX\\2016.11\\Win_x64\\EXE\\mx201611.exe',
                '-f','mxspe009_'+str(step)+"_perm_"+str(i+1)+'.dat'])
            subprocess.run(['C:\\Program Files (x86)\\CMG\\BR\\2016.11\\Win_x64\\EXE\\report.exe',
                        '-f', 'mxspe009_Production.rwd', '-o', 'mxspe009_Production.rwo'])      
            subprocess.run(['C:\\Program Files (x86)\\CMG\\BR\\2016.11\\Win_x64\\EXE\\report.exe',
                        '-f', 'mxspe009_Pressure.rwd', '-o', 'mxspe009_Pressure.rwo'])
        
    
            if os.path.getsize('mxspe009_Pressure.rwo')==0 or os.path.getsize('mxspe009_Production.rwo')==0:
                time.sleep(300)    
                subprocess.run(['C:\\Program Files (x86)\\CMG\\BR\\2016.11\\Win_x64\\EXE\\report.exe',
                    '-f', 'mxspe009_Production.rwd', '-o', 'mxspe009_Production.rwo'])      
                subprocess.run(['C:\\Program Files (x86)\\CMG\\BR\\2016.11\\Win_x64\\EXE\\report.exe',
                    '-f', 'mxspe009_Pressure.rwd', '-o', 'mxspe009_Pressure.rwo'])
            
            pressure_data=pd.read_csv('mxspe009_Pressure.rwo',delim_whitespace=True,skiprows=2,header=None)
            pressures[i]=(pressure_data.values.flatten()[:375])
            next_production,reward=calculate_reward(step+1)
            productions[i]=next_production
            rewards[i]=reward             
            
        avg_pressure=np.mean(pressures,axis=0)    
        avg_pressure=np.reshape(avg_pressure,(5,5,15),order='F') 
        
        avg_production=np.mean(productions,axis=0)
        
        avg_reward=np.mean(rewards,axis=0)          
   
      
        return avg_pressure, avg_production,avg_reward 

    def reset(self):
        
        pressures=np.zeros((50,375))

        for i in range(50):

            subprocess.run(['C:\\Program Files (x86)\\CMG\\IMEX\\2016.11\\Win_x64\\EXE\\mx201611.exe',
                '-f','mxspe009_1_perm_'+str(i+1)+'.dat'])
                        
            subprocess.run(['C:\\Program Files (x86)\\CMG\\IMEX\\2016.11\\Win_x64\\EXE\\mx201611.exe',
                '-f','mxspe009_2_perm_'+str(i+1)+'.dat'])
                        
            with open('mxspe009_Pressure.rwd','w') as f:
                file_data=["*FILES 'mxspe009_2_perm_"+str(i+1)+".irf'\n", '\n', "*PROPERTY-FOR 'Pressure' 180 *SRF-FORMAT\n"]
                f.writelines(file_data)
                f.close 
    
            subprocess.run(['C:\\Program Files (x86)\\CMG\\BR\\2016.11\\Win_x64\\EXE\\report.exe',
                        '-f', 'mxspe009_Pressure.rwd', '-o', 'mxspe009_Pressure.rwo'])
                       
            pressure_data=pd.read_csv('mxspe009_Pressure.rwo',delim_whitespace=True,skiprows=2,header=None)
            pressures[i]=(pressure_data.values.flatten()[:375])
            
        avg_pressure=np.mean(pressures,axis=0)
        avg_pressure=np.reshape(avg_pressure,(5,5,15),order='F')

        with open('SCHEDUEL.txt','w') as f:    
        
            file_data=['*TIME 1\n','*DTWELL 1\n',"OPEN   'Injector'\n", '*TIME 90\n',"OPEN   'Injector'\n",
                       'WRST TIME\n','WPRN WELL TIME\n','WSRF GRID TIME\n','*TIME 180\n','*STOP\n']           
            file_data=file_data+[]
            f.writelines(file_data)
            f.close
                    
        return avg_pressure,np.array([0,0,0])
    
if __name__=='__main__':
    env=generateEnv()
    env.reset()    
    action=[1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    total_reward=0
    for i in range(3,19):

        _,_,reward=env.update(action=action[i-1],step=i)
        total_reward=total_reward+reward
        print(reward)
    print(total_reward)
#    data=pd.read_csv('mxspe009_Pressure_Initial.rwo',delim_whitespace=True,skiprows=2,header=None)
#    pressure=data.values.flatten()[:375]    
#    print(pressure.mean())
    
    