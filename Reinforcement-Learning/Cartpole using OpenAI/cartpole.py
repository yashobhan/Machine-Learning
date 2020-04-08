import gym

import random
import numpy as np 
import pandas as pd 

from statistics import mean
from statistics import median
from collections import Counter 

# import tflearn
# from tflearn.layers.core import input_data
# from tflearn.layers.core import dropout
# from tflearn.layers.core import fully_connected
# from tflearn.layers.estimator import regression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

ETA = 1e-2
dropout_rate = 0.8

env = gym.make('CartPole-v0')

goal_steps = 500 # Tweak later
score_requirement = 50 # Tweak later
initial_games = 100000 # Tweak later


def random_games():
    for episode in range(1):
        env.reset()
        for t in range(goal_steps):
            # env.render() # Comment to not render
            action = env.action_space.sample() # Take a rando action in the environment (Works in any env)
            observation, reward, done, info = env.step(action)
            if done:
                break
# random_games()


def initial_population():
    # Only append training data if score happens to be above 50
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []

        env.reset()

        for _ in range(goal_steps):
            action = random.randrange(0, 2) # Can also use sample
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            previous_observation = observation
            score += reward

            if done:
                break


        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])
        
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('data.npy', training_data_save)

    print('Average Accepted Score: ', mean(accepted_scores))
    print('Median Accepted Score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data



# def neural_network(input_size, dropout_rate):
#     network = input_data(shape=[None, input_size, 1], name='input')

#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     network = fully_connected(network, 512, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     netowrk = fully_connected(network, 1024, activation='relu')
#     network = dropout(network, dropout_rate)

#     network = fully_connected(network, 512, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, dropout_rate) # dropout_rate is keep rate rather than dropout rate wtf

#     network = fully_connected(network, 2, activation='softmax')
#     network = regression(network, optimizer='adam', learning_rate=ETA, loss='categorical_crossentropy', name='outputs')

#     model = tflearn.DNN(network)

#     return model


def keras_net(input_size):
    model = Sequential()

    model.add(Dense(128, input_shape=(None, input_size, 1), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))

    # # model.add(Dense(256, activation='relu'))
    # # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))

    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(lr=ETA)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model
    



def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) # [X - Observations, Outputs]
    print('First X: ', X[0])
    print('Shape of X: ', X.shape)
    
    y = np.array([i[1] for i in training_data]) # [Observations, y - Outputs]
    print('Initial y: ', y[0])
    print('Shape of y: ', y.shape)

    if not model:
        model = keras_net(input_size=len(X[0]))

    # model.fit(X, y, epochs=5, show_metric=True, run_id='openAI')
    model.fit(X, y, epochs=5, verbose=1)

    return model



training_data = initial_population()
model = train_model(training_data)



choices = []
scores = []

for each_game in range(10):
    score = 0
    game_memory = []
    previous_observation = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if len(previous_observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(previous_observation.reshape(-1, len(previous_observation), 1)) [0])
            # print('Action: ', action)
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        previous_observation = new_observation
        game_memory.append([new_observation, action])
        score += reward
        # print('Score: ', score)

        if done:
            break
        
    scores.append(score)


print('Average Score: ', mean(scores))