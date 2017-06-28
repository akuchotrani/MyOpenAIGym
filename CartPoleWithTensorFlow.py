# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:13:24 2017

@author: aakash.chotrani
"""

import gym
#from gym import wrappers
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

LearningRate = 0.1
LR = 1e-3
env = gym.make('CartPole-v0')

env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 1000

#playing some random game and just rendereing it to see the lunar lander fucking crash!!!------oh yes!
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


#initial_population()

#neural network takes an input size becuase we can save our model and train our neural net based on that
#this funtion will just return the model
def neural_network_model(input_size):
    
    network = input_data(shape=[None,input_size,1],name = 'input')
    
    #there are 128,256,512 nodes at 5 layers and the activation is rectified linear 'relu'
    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.8)
    
    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.8)
    
    network = fully_connected(network,512,activation = 'relu')
    network = dropout(network,0.8)
    
    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.8)
    
    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.8)
    
    #output layer it takes 2 outputs
    network = fully_connected(network,2,activation = 'softmax')
    network = regression(network,optimizer = 'adam', learning_rate = LR,
                         loss = 'categorical_crossentropy', name = 'targets')
    
    model = tflearn.DNN(network,tensorboard_dir = 'newLog')
    return model
    
#Now let's train our model taking the training data and our model which we created in previous function:-
#by default we are setting the model to false if we don't have a model it will create a model for us  
def train_model(training_data,model = False):
    #we waant to save the observation from training data which contains observation and action it took
    #print('training data observarion length: ',len(training_data[0][0]))
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y = [i[1] for i in training_data]
    #print('X: ',X)
    if not model:
        model = neural_network_model(input_size = len(X[0]))
        
    model.fit({'input': X},{'targets':Y},n_epoch = 3, snapshot_step = 500, show_metric = True, run_id = 'openaistuff')
    
    return model


training_data = initial_population()
model = train_model(training_data)
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1',force = True)

#model.save('aakashFirstModel.model')

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        #for the first time step we don't know which move to make hence take random action
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        #just to see which choices our neural network makes 
        choices.append(action)
        
        new_observation,reward,done,info = env.step(action)
        prev_obs = new_observation
        #keep saving the fucking game to retrain data
        game_memory.append([new_observation,action])
        score += reward
        if done:
            break
    scores.append(score)
    
    
print('Average Score: ',sum(scores)/len(scores))
print('Choice1: {}, Choices2: {}'.format(choices.count(1)/len(choices),
      choices.count(0)/len(choices)))
env.close()
#gym.upload('/tmp/cartpole-experiment-1', api_key='sk_Y7AhINO1TGGWXW7XCEhCzw')

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            