import numpy as np
import random
import matplotlib.pyplot as plt
y1=[0]*1000
y2=[0]*1000
y3=[0]*1000
#initialising the y coordinates for the 3 graphs 
def reward(e):
    reward1=0
    reward2=0
    reward3=0
    reward4=0
    reward5=0
    #initialising all rewards pertaining to each action as 0
    freq1=0
    freq2=0
    freq3=0
    freq4=0
    freq5=0
    #initialising the frequency of each action as 0
    value1=0
    value2=0
    value3=0
    value4=0
    value5=0
    #initialising all action values to 0
    #so that an unexplored action is chosen instead of an action with a negative action value
    reward=[0]*100
    #initialising the array that contains the reqards of each time step
    for i in range(100):
        if i==1:
            k=np.random.randint(1,6)
            #starting with a random action
            if k==1:
                reward[i]=np.random.default_rng().normal(2,1,1)[0]
                reward1+=reward[i]
                freq1+=1
                value1=reward1/freq1
                #normal distribution simulation
            if k==2:
                l=np.random.randint(0,2)
                if l==1:
                    reward[i]=5
                    reward2+=reward[i]
                
                else:
                    reward[i]=-6
                    reward2+=reward[i]
                freq2+=1
                value2=reward2/freq2
                #coin flip simulation
            if k==3:
                reward[i]=np.random.default_rng().poisson(2,1)[0]
                reward3+=reward[i]
                freq3+=1
                value3=reward3/freq3
                #poisson distribution simulation
            if k==4:
                reward[i]=np.random.default_rng().exponential(3,1)[0]
                reward4+=reward[i]
                freq4+=1
                value4=reward4/freq4
                #exponential distribution simulation
            if k==5:
                n=np.random.randint(1,5)
                if n==1:
                    reward[i]=np.random.default_rng().normal(2,1,1)[0]
            
                if n==2:
                    l=np.random.randint(0,2)
                    if l==1:
                        reward[i]=5
            
                    else:
                        reward[i]=-6
            
                if n==3:
                    reward[i]=np.random.default_rng().poisson(2,1)[0]

                if n==4:
                    reward[i]=np.random.default_rng().exponential(3,1)[0]
                reward5+=reward[i]
                freq5+=1
                value5=reward5/freq5
                #simulating the crazy button
        else:
            if e==0:
                j=1
                #this makes sure that the action with the highest value is always picked
                #eliminates the case where a random action is picked
            else:
                j=np.random.randint(0,1/e)
                #adjusting probability of picking a random action(exploration)
            if j==0:
                k=np.random.randint(1,6)
                #random action is picked
                #same as step 1 effectively
                if k==1:
                    reward[i]=np.random.default_rng().normal(2,1,1)[0]
                    reward1+=reward[i]
                    freq1+=1
                    value1=reward1/freq1
                if k==2:
                    l=np.random.randint(0,2)
                    if l==1:
                        reward[i]=5
                        reward2+=reward[i]
                
                    else:
                        reward[i]=-6
                        reward2+=reward[i]
                    freq2+=1
                    value2=reward2/freq2
                if k==3:
                    reward[i]=np.random.default_rng().poisson(2,1)[0]
                    reward3+=reward[i]
                    freq3+=1
                    value3=reward3/freq3
                if k==4:
                    reward[i]=np.random.default_rng().exponential(3,1)[0]
                    reward4+=reward[i]
                    freq4+=1
                    value4=reward4/freq4
                if k==5:
                    n=np.random.randint(1,5)
                    if n==1:
                        reward[i]=np.random.default_rng().normal(2,1,1)[0]
            
                    if n==2:
                        l=np.random.randint(0,2)
                        if l==1:
                            reward[i]=5
            
                        else:
                            reward[i]=-6
            
                    if n==3:
                        reward[i]=np.random.default_rng().poisson(2,1)[0]

                    if n==4:
                        reward[i]=np.random.default_rng().exponential(3,1)[0]
                    reward5+=reward[i]
                    freq5+=1
                    value5=reward5/freq5
            else:
                g=max(value1,value2,value3,value4,value5)
                #making sure the action with highest value is picked
                if g==value1:
                    reward[i]=np.random.default_rng().normal(2,1,1)[0]
                    reward1+=reward[i]
                    freq1+=1
                    value1=reward1/freq1
                elif g==value2:
                    l=np.random.randint(0,2)
                    if l==1:
                        reward[i]=5
                        reward2+=reward[i]
                
                    else:
                        reward[i]=-6
                        reward2+=reward[i]
                    freq2+=1
                    value2=reward2/freq2
                elif g==value3:
                    reward[i]=np.random.default_rng().poisson(2,1)[0]
                    reward3+=reward[i]
                    freq3+=1
                    value3=reward3/freq3
                elif g==value4:
                    reward[i]=np.random.default_rng().exponential(3,1)[0]
                    reward4+=reward[i]
                    freq4+=1
                    value4=reward4/freq4
                elif g==value5:
                    n=np.random.randint(1,5)
                    if n==1:
                        reward[i]=np.random.default_rng().normal(2,1,1)[0]
            
                    if n==2:
                        l=np.random.randint(0,2)
                        if l==1:
                            reward[i]=5
            
                        else:
                            reward[i]=-6
            
                    if n==3:
                        reward[i]=np.random.default_rng().poisson(2,1)[0]

                    if n==4:
                        reward[i]=np.random.default_rng().exponential(3,1)[0]
                    reward5+=reward[i]
                    freq5+=1
                    value5=reward5/freq5
    #sending the final array which contains the rewards obtianed at each time step as output                
    return reward
x=[i+1 for i in range(1000)]
#initialising the x axis for the graph
for i in range (1000):
    a=reward(0)
    for i in range (100):
        y1[i]+=a[i]/1000
for i in range (1000):
    a=reward(0.01)
    for i in range (100):
        y2[i]+=a[i]/1000
for i in range (1000):
    a=reward(0.1)
    for i in range (100):
        y3[i]+=a[i]/1000

#here for each of the 3 values of e we have made an array whose ith element 
#contains the average of the rewards obtained in the ith timestep over 1000 episodes all with the same value of e

plt.plot(x,y1,label="e=0")
plt.plot(x,y2,label="e=0.01")
plt.plot(x,y3,label="e=0.1")
plt.legend()
plt.show()
#plotting the 3 graphs with appropriate labelling
        

















































































































































































































































































































































            




            
























            







            