# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:45:09 2017

@author: aakash.chotrani
"""

import gym
#from gym import wrappers

env = gym.make('LunarLander-v2')
#env = wrappers.Monitor(env, "/tmp/gym-results")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        print("xPosition: ",observation[0],"yPosition: ",observation[1])
        env.reset()

env.close()
#gym.upload("/tmp/gym-results", api_key="sk_Y7AhINO1TGGWXW7XCEhCzw")