# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:58:15 2018

@author: NAZMUL
"""

import gym
import numpy as np
import matplotlib as plt
env_name = 'Breakout-v4'
env = gym.make(env_name)
obs=env.reset()
print(obs.shape)
def random_policy(n):
    action=np.random.randint(0,n)
    return action
for step in range(1000):
    action = random_policy(env.action_space.n)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        img = env.render(mode='rgb_array')
        plt.imshow(img)
        plt.show()
        print("That game is over in {} steps".format(step))
        break