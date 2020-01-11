import os
import sys
import time
import io
from io import BytesIO

import cv2
from PIL import Image

import numpy as np 
import pandas as pd
from random import randint

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

#Keras baby!
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from collections import deque

import random
import pickle
import base64
import json

#PATH VARIABLES
QVALUES_FILE = 'knowledge/qvalues_df.csv'
GAME_URL = 'chrome://dino'
CHROME_DRIVER = '../chromedriver'
LOSS_FILE_PATH = 'knowledge/loss_df.csv'
ACTIONS_FILE_PATH = 'knowledge/actions_df.csv'
SCORES_FILE_PATH = 'knowledge/scores_df.csv'

WEB_DRIVER_PATH = 'D:\\Projects\\Python\\Chrome Dino\\chromedriver_win32\\chromedriver.exe'


INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

#Getting image from canvash
getBase64Image = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

MODEL_NAME = 'experience.h5'

loss_df = pd.read_csv(LOSS_FILE_PATH) if os.path.isfile(LOSS_FILE_PATH) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(SCORES_FILE_PATH) if os.path.isfile(SCORES_FILE_PATH) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(ACTIONS_FILE_PATH) if os.path.isfile(ACTIONS_FILE_PATH) else pd.DataFrame(columns=['actions'])
q_values_df = pd.read_csv(QVALUES_FILE) if os.path.isfile(QVALUES_FILE) else pd.DataFrame(columns=['qvalues'])




#Game class will create an interface between Python and Chrome
class Game:

	#__init__(): Will launch the browser window using the attributes in chrome_options
	def __init__(self, custom_config=True):
		chrome_options = Options()
		chrome_options.add_argument("disable-infobars")
		chrome_options.add_argument('--mute-audio')

		self._driver = webdriver.Chrome(WEB_DRIVER_PATH, chrome_options=chrome_options)
		self._driver.set_window_position(x=-10, y=0)
		self._driver.get('chrome://dino')
		self._driver.execute_script("Runner.config.ACCELERATION=0")
		self._driver.execute_script(INIT_SCRIPT)

		print('-------- __init__ works!---------')

	#get_crashed(): returns True if agent crashes into a Tree	
	def get_crashed(self):
		return self._driver.execute_script("return Runner.instance_.crashed")
		print('------------ Game Crashed ------------')


	#get_playing(): True if game in progress, False if game crashed etc.
	def get_playing(self):
		return self._driver.execute_script("return Runner.instance_.playing")

	#restart(): Restarts the game after a crash
	def restart(self):
		self._driver.execute_script("Runner.instance_.restart()")

	#press_up(): Action UP
	def press_up(self):
		self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

	#press_down(): Action DOWN
	def press_down(self):
		self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

	#get_score(): Returns current score
	def get_score(self):
		score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
		score = ''.join(score_array) #Object = Array, score format is [1, 0, 0] ie 100
		return int(score)

	#pause(): Pauses the game
	def pause(self):
		return self._driver.execute_script("return Runner.instance_.stop()")

	#resume(): Resumes game after pausing
	def resume(self):
		return self._driver.execute_script("return Runner.instance_.play()")

	#end(): End session
	def end(self):
		self._driver.close()



#DinoAgent class will manipulate the agent
class DinoAgent:
	def __init__(self, game):
		self._game = game
		self.jump() #Jump once inititally to start the game

	def is_running(self):
		return self._game.get_playing()

	def is_crashed(self):
		return self._game.get_crashed()

	def jump(self):
		self._game.press_up()

	def duck(self):
		self._game.press_down()


#Ops on knowledge
def save_obj(obj, name):
	with open('knowledge/' + name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) #Dump files into Pickle '/knowledge'

def load_obj(name):
	with open('knowledge/' + name + '.pkl', 'rb') as f:
		return pickle.load(f) #Reloading pickle dump

def grab_screen(_driver):
	image_b64 = _driver.execute_script(getBase64Image)
	screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
	image = preprocess_image(screen)
	return image

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image[:300, :500] #Crop ROI
	image = cv2.resize(image, (80, 80))
	return image

def show_img(graphs=False):
	#Show Images in new Window
	while True:
		screen = (yield)
		window_title = "logs" if graphs else "game_play"
		imS = cv2.resize(screen, (800, 400))
		cv2.imshow(window_title, screen)
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			cv2.destroyAllWindows()
			break




#Manipulating Game State
class Game_State:
	def __init__(self, agent, game):
		self._agent = agent
		self._game = game 
		self._display = show_img() #Display processed image using OpenCV
		self._display.__next__() #Display Coroutine

	def get_state(self, actions):
		actions_df.loc[len(actions_df)] = actions[1] #Storing actions in a DataFrame
		score = self._game.get_score()
		reward = 0.1
		is_over = False #Game Over!

		if actions[1] == 1:
			self._agent.jump()

		image = grab_screen(self._game._driver)
		self._display.send(image) #Display image on Screen

		if self._agent.is_crashed():
			scores_df.loc[len(loss_df)] = score #Log score when Game Over
			self._game.restart()
			reward = -1
			is_over = True

		return image, reward, is_over



#Init log structures from files if exists
#Else create new






#Main Game Parameters
ACTIONS = 2 #Possible ACTIONS: Jump, Do Nothing
GAMMA = 0.99 #Decay rate of past observations
OBSERVATION = 100 #Timesteps to observe before training
EXPLORE = 100000 #Frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 #Final value of epsilon
INITIAL_EPSILON = 0.1 #Starting value of epsilon
PAST_MEMORY = 2000 #Number of previous transitions to remember
BATCH = 64
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-3	
IMG_ROWS, IMG_COLS = 80, 80
IMG_CHANNELS = 4


#Save training values as Checkpoints to resume training
def init_cache():
	save_obj(INITIAL_EPSILON, "epsilon")
	t = 0
	save_obj(t, "time")
	D = deque()
	save_obj(D, "D")


#Call once to init file Structure

def NeuralNet():
	print('Initializing the Neural Network')
	model = Sequential()
	model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
	# model.add(Conv2D(32, (8, 8), strides=(2, 2), padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
	# model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
	# model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(ACTIONS))
	adam = Adam(lr=LEARNING_RATE)
	model.compile(loss='mse', optimizer=adam)


	#Create Model if not present
	if not os.path.exists(LOSS_FILE_PATH):
		model.save_weights(MODEL_NAME)
	print('---------- Neural Net building complete ----------')
	return model


def trainNetwork(model, game_state, observe=False):
	last_time = time.time()
	D = load_obj("D")

	#Get first state by doing nothing
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1 #0 => Do nothing, 1 => Jump

	x_t, r_0, terminal = game_state.get_state(do_nothing)
	s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) #Stack four images to create placeholder input

	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*20*20*4 Dim Ordering

	initial_state = s_t

	if observe:
		OBSERVE = 999999999 #Keep observing, never learn
		epsilon = FINAL_EPSILON
		print('Loading Weights')
		model.load_weights(MODEL_NAME)
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
		print('Model Loaded Successfully')

	else:
		OBSERVE = OBSERVATION
		epsilon = FINAL_EPSILON
		model.load_weights(MODEL_NAME)
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)

	t = load_obj("time") #Resume from the previous time step stored in the logs
	
	while(True):
		loss = 0
		Q_sa = 0
		action_index = 0
		r_t = 0 #Reward 0, first one at 4
		a_t = np.zeros([ACTIONS])

		if t % FRAME_PER_ACTION == 0:
			if random.random() <= epsilon:
				#Randomly explore an action
				print('--------------- Random Action-----------------')
				action_index = random.randrange(ACTIONS)
				a_t[action_index] = 1

			else:
				q = model.predict(s_t) #Prediction over a stack of 4 images
				max_Q = np.argmax(q)
				action_index = max_Q
				a_t[action_index] = 1 #0 - do nothing, 1 - jump

		#Reducing the exploration parameter (Epsilon) gradually
		if epsilon > FINAL_EPSILON and t > OBSERVE:
			epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		#Run selcted action, observed next state and reward
		x_t1, r_t, terminal = game_state.get_state(a_t)
		print('FPS: {0}'.format(round(1 / (time.time()-last_time)), 0)) #Framerate
		last_time = time.time()
		x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
		s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #Append new image and remove old one


		D.append((s_t, action_index, r_t, s_t1, terminal))
		if len(D) > PAST_MEMORY:
			D.popleft()
		

		#Only train if done observing
		if t > OBSERVE:

			minibatch = random.sample(D, BATCH)
			inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
			targets = np.zeros((inputs.shape[0], ACTIONS))

			#The bot will experience his past and learn from it here
			for i in range(0, len(minibatch)):
				state_t = minibatch[i][0] #4D stacks of images
				action_t = minibatch[i][1]
				reward_t = minibatch[i][2] #Reward at state state_t due to action action_t
				state_t1 = minibatch[i][3] #Next action
				terminal = minibatch[i][4] #Whether the bot survived or died due to his actions


				inputs[i:i + 1] = state_t

				targets[i] = model.predict(state_t) #Predict q values
				Q_sa = model.predict(state_t1)

				if terminal:
					targets[i, action_t] = reward_t
				else:
					targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

			loss += model.train_on_batch(inputs, targets)
			loss_df.loc[len(loss_df)] = loss
			q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
		s_t = initial_state if terminal else s_t1 
		t = t + 1

		#Save every 1000 iterations
		if t % 1000 == 0:
			game_state._game.pause() #pause game while saving to filesystem
			model.save_weights(MODEL_NAME, overwrite=True)
			save_obj(D,"D") #saving episodes
			save_obj(t,"time") #caching time steps
			save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
			loss_df.to_csv("./knowledge/loss_df.csv",index=False)
			scores_df.to_csv("./knowledge/scores_df.csv",index=False)
			actions_df.to_csv("./knowledge/actions_df.csv",index=False)
			q_values_df.to_csv(QVALUES_FILE,index=False)
			with open("model.json", "w") as outfile:
				json.dump(model.to_json(), outfile)
			game_state._game.resume()

		state = ""

		if t <= OBSERVE:
			state = "observe"
		elif t > OBSERVE and t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print("TIMESTEP", t, "| STATE", state, "|EPSILON", epsilon, "| ACTION", action_index, "| REWARD", r_t, "| Q_MAX " , round(np.max(Q_sa), 4), "| Loss ", round(loss, 4))
	print("Episode finished!")
	print("************************")


def playGame(observe=False):
	game = Game()
	dino = DinoAgent(game)
	game_state = Game_State(dino, game)
	model = NeuralNet()

	try:
		trainNetwork(model, game_state, observe=observe)
	except StopIteration:
		game.end()


# init_cache()
playGame(observe=False)
