# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:13:24 2017

@author: aakash.chotrani
"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

LearningRate = 0.1
env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info  = env.step(action)
            
            if done:
                print(observation)
                break
        
        
#some_random_games()


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    
    #iterating through 10000 games 
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        
        #iterating through actual game in 500 frames
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation,reward,done,info = env.step(action)
            
             #based on the previous observation we record the current action
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
            
            #saving the current observation into memory for next frame
            prev_observation = observation
            score += reward
            if done:
                break
        #if the game was good then we store the gamedata into game_memory
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
            
            training_data.append([data[0],output])
        env.reset()
        scores.append(score)
    
    #converting training data into numpy array and saving it
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    print('Average accepted score: ',mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data


initial_population()
                