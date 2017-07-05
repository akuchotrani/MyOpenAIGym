# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:45:09 2017

@author: aakash.chotrani
"""

import gym
import numpy as np
from statistics import mean,median
from collections import Counter   #	dict subclass for counting hashable objects
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


env = gym.make('LunarLander-v2')
observation = env.reset()

LR = 1e-3
LearningRate = 0.00001

env.reset()

goal_steps = 500
score_requirement = 0
initial_games = 3000


def play_some_random_games():
    
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            print("\n random actions: ",action)
            observation, reward, done, info  = env.step(action)
            
            if done:
                print(observation)
                break

#play_some_random_games()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    good_games = 0
    print("Total Actions possible: ",env.action_space.n)
    #iterating through 10000 games 
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        
        #iterating through actual game in 500 frames
        for _ in range(goal_steps):
            action = env.action_space.sample()
            observation,reward,done,info = env.step(action)
            
             #based on the previous observation we record the current action
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
                #print("\ngame_memory: ",game_memory)
            #saving the current observation into memory for next frame
            prev_observation = observation
            score += reward
            if done:
                break
            
        #if the game was good then we store the gamedata into game_memory
        #print("\nScore: ",score)
        if score >= score_requirement:
            good_games = good_games + 1
            accepted_scores.append(score)
            for data in game_memory:
                
                if data[1] == 0:
                    output = [1,0,0,0]
                elif data[1] == 1:
                    output = [0,1,0,0]
                elif data[1] == 2:
                    output = [0,0,1,0]
                elif data[1] == 3:
                    output = [0,0,0,1]
            
                training_data.append([data[0],output])
        env.reset()
        scores.append(score)
    
    #converting training data into numpy array and saving it
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    print('Average accepted score: ',mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    print("Good_games: ",good_games)
    return training_data

#initial_population()


#neural network takes an input size becuase we can save our model and train our neural net based on that
#this funtion will just return the model
def neural_network_model(input_size):
    
    #input_size =  8 ,we have to take into account 8 values returned by our observation
    #[Xposition,Yposition,Xvelocity,Yvelocity,landerAngle,angularVelocity,rightLeg,leftLeg]
    network = input_data(shape=[None,input_size,1],name = 'input')
    
    #there are 128,256,512 nodes at 5 layers and the activation is rectified linear 'relu'
    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.5)
    
    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.5)
    
    network = fully_connected(network,512,activation = 'relu')
    network = dropout(network,0.5)
    
    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.5)
    
    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.5)
    
    #output layer takes 2 outputs
    network = fully_connected(network,4,activation = 'sigmoid')
    network = regression(network,optimizer = 'sgd', learning_rate = LR,
                         loss = 'categorical_crossentropy', name = 'targets')
    
    #model = tflearn.DNN(network,tensorboard_dir = 'newLog')
    model = tflearn.DNN(network,tensorboard_verbose = 3)
    return model


#Now let's train our model taking the training data and our model which we created in previous function:-
#by default we are setting the model to false if we don't have a model it will create a model for us  
def train_model(training_data,model = False):
    #we waant to save the observation from training data which contains observation and action it took
    print('training data observarion length: ',len(training_data[0][0]))
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    print('\n\n X = ',X)
    Y = [i[1] for i in training_data]
    print('Y = ',Y)
    #print('X: ',X)
    if not model:
        print("len X[0]: ",len(X[0]))
        model = neural_network_model(input_size = len(X[0]))
        
    model.fit({'input': X},{'targets':Y},n_epoch = 3, snapshot_step = 500, show_metric = True, run_id = 'openaistuff')
    
    return model

training_data = initial_population()
model = train_model(training_data)


scores = []
choices = []

for each_game in range(20):
    score = 0
    game_memory = []
    prev_obs = []
    lol = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        #for the first time step we don't know which move to make hence take random action
        if len(prev_obs) == 0:
            action = env.action_space.sample()
            print("hi first frame take random action")
        else:
            #lol = prev_obs.reshape(-1,len(prev_obs),1)[0]
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            
            #print("\n\nMy predicted Action:-")
           # print("\nprev_obs: ",lol)
           # print("\n predict: ",model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            #print("\n argmax: ",action)
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
#print('Choice1: {}, Choices2: {}'.format(choices.count(1)/len(choices),
#      choices.count(0)/len(choices)))
env.close()