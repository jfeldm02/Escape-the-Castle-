import time
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial 
	   configuration and taking actions until a terminal state is reached.
	2. Keep track of gameplay history in an appropriate format for each of the episodes.
	3. From gameplay history, estimate the probability of victory against each of the guards when taking the fight action.

	Some important notes:

		a. Keep in mind that given some observation [(X,Y), health, guard_in_cell], a fight action is only meaningful if the 
		   last entry corresponding to guard_in_cell is nonzero.

		b. Upon taking the fight action, if the player defeats the guard, the player is moved to a random neighboring cell with 
		   UNCHANGED health. (2 = Full, 1 = Injured, 0 = Critical).

		c. If the player loses the fight, the player is still moved to a random neighboring cell, but the health decreases by 1.

		d. Your player might encounter the same guard in different cells in different episodes.

		e. All interaction with the environment must be done using the env.step() method, which returns the next
		   observation, reward, done (Bool indicating whether terminal state reached) and info. This method should be called as 
		   obs, reward, done, info = env.step(action), where action is an integer representing the action to be taken.

		f. The env.reset() method resets the environment to the initial configuration and returns the initial observation. 
		   Do not forget to also update obs with the initial configuration returned by env.reset().

		g. To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		   For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		   will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		h. To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		   Example usage below. This function should be called after every action.

		   if gui_flag:
		       refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the np array, P which contains four float values, each representing the probability of defeating guards 1-4 respectively.

'''
def update(action):
	obs, reward, done, info = env.step(action)
	if gui_flag:
		refresh(obs, reward, done, info) # Update the game screen [GUI only]
	#print(f"Obs:", obs)
	return obs, reward, done, info

def estimate_victory_probability(num_episodes= 100000): #100000

	P = np.zeros(len(env.guards))
	successFight = np.zeros(len(env.guards), dtype = int)
	fightEvent = np.zeros(len(env.guards), dtype = int)
	#guard_location = np.zeros(len(env.guards))
	#iter = 0

	for iter in range(num_episodes):
		
		print(iter)
		env.reset() # initialize game from scratch
		obs, reward, done, info = update('HIDE')

		while not done: # while episode is still ongoing

			# Move randomly while a guard isn't found :
			if obs['guard_in_cell'] is None:
				rand_action = random.choice(env.actions[:4]) 
				obs, reward, done, info = update(rand_action) # take observations of new state and refresh GUI

			# Guard is found! 
			elif obs['guard_in_cell'] is not None:
				guard_map = {'G1': 0,'G2': 1,'G3': 2,'G4': 3} 
				guard = guard_map[obs['guard_in_cell']]
				rewardBefore = reward # Getting a value to compare hide/fight reward result to 
				obs, reward, done, info = update('FIGHT') # Fight the guard right away. This function is for the probability of winning in a fight only
			
				# Updating P with each potential outcome of the actions 
				if reward > rewardBefore: # Forcing a fight, and fighting was a success
					successFight[guard] += 1 
					fightEvent[guard] += 1
			
				elif reward < rewardBefore: # Forcing a fight, and fighting was a failure 
					fightEvent[guard] += 1 
					
	for guard in range(len(successFight)): 
		P[guard] = successFight[guard]/fightEvent[guard]  
	return P

t0 = time.time()
P = estimate_victory_probability()
actual = [0.2, 0.4, 0.1, 0.3]
print(f"Probability of beating each guard:", P)
print(f'Probability of beating the guards should be:', actual)
t1 = time.time()
howlong = t1-t0
print(f'Took this much time:', howlong)

