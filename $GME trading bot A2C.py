#!/usr/bin/env python
# coding: utf-8

# In[1]:


#gym for the environments
import gym
import gym_anytrading
#for the reinforcement learning algorithms
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
#for general usage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("/home/ramanuja/Documents/gmedata.csv")


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df["Date"] = pd.to_datetime(df["Date"])


# In[6]:


df.dtypes


# In[7]:


df.set_index("Date",inplace=True)


# In[8]:


df=df.sort_index()
df.head()


# In[9]:


env = gym.make("stocks-v0",df=df,frame_bound=(5,100),window_size=5)


# In[10]:


env.prices


# In[11]:


env.signal_features


# In[12]:


state =env.reset()
while True:
    action = env.action_space.sample()
    n_state,reward,done,info = env.step(action)
    if done:
        print("info",info)
        break
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


# In[13]:


env_maker = lambda: gym.make("stocks-v0",df=df,frame_bound=(5,100),window_size=5)
env = DummyVecEnv([env_maker])


# In[14]:


model = A2C("MlpLstmPolicy",env,verbose=1)
model.learn(total_timesteps=100000)


# In[15]:


env = gym.make("stocks-v0",df=df,frame_bound=(90,110),window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis,...]
    action,_states = model.predict(obs)
    obs,rewards,done,info = env.step(action)
    if done:
        print(info)
        break


# In[ ]:




