#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import gym
from gym import spaces
from collections import deque
import sys
py_script = "/Users/jbrown/OneDrive - Queen's University Belfast/python_scripts"
if py_script is not sys.path:
	sys.path.append(py_script)	 
from ctap_evolution import CTAP



class ControlEnv:
	# Because of google colab, we cannot implement the GUI ('human' render mode)
	metadata = {'render.modes': ['console']}
	
	def __init__(self, num_levels, omega_max, omega_multiplier, max_time, timestep, alpha, beta, detuning, 
					detuning_fixed, detuning_percent, dephasing, dephasing_sigma, 
					observation_verbosity, recall_len):
		""" A class of Control environments used in simulating the two barrier control pulse
			in the Stirap/Ctap regimes.
			
		Args:
			num_levels (int): 	Defines the number of energy levels/ accesible quantum states 
							  	of the system. 3 by default, limited functionality for 5.
			omega_max (float):	 The maximum value that can be taken by the two pulses.
			omega_multiplier (list, optional):  Determines the ratio between maximum 
												possible pulse strengths of the two control 
												pulses.
			max_time (int): 	Maximum number of timeSTEPS for an episode.
			timestep (int): 	The length of each individual timestep. (Determines for how
								long we evolve the system under that portion of the piecewise
								constant hamiltonian) 
			alpha (float): 		Multiplier for the population transfer term in the reward 
								fuunction.
			beta (float): 		Multiplier for the punishing term in the reward function.
			detuning (bool): 	Whether to include detuning in the system. Defaults to False.
			detuning_fixed (bool): 	Whether said detuning is to be a fixed percentage of the maximum
									pulse value or stochastically sampled from a range of values at
									each timestep.
			detuning_percent (float): 	Percent of fixed detuning or extent of range for stochastic.
			dephasing (bool): 		Whether to include dephasing in the dynamics. Defaults to False.
			dephasing_sigma (float):  dephasing strength as a fraction (of omega_max).
			observation_+verbosity (int): 	Verbosity of the observation passed to the neural net.
											0: Diagonal elements only, 1: Diagonal elements and 
											previous observations, 2: full density matrix, 3: Full 
											density matrix and previous observations.
		
		"""
		# Defines the number of quantum dots. Will usually be 3 and will ALWAYS be odd.	
		# Then define the number of inter-dot couplings
		self.num_levels = num_levels
		self.num_couplings = (num_levels - 1)
		
		# Maximum possible value of the barrier control gate pulses
		self.omega_max = omega_max
		# np.array that determines the ratio between the max values of the two pulses. default to [1,1]
		self.omega_multiplier = np.array(omega_multiplier)
		# Maximum number of timesteps for each individual environmental episode
		self.max_time = max_time
		# The size of each time step in an episode. This dictates for how long QuTIP will
		# evolve the state for that step in the MDP
		self.timestep = timestep
		
		# Parameters relating to the reward function. alpha encodes the importance of 
		# of transferring the population to the final state, and beta is a multiplier
		# for the punishing term for occupation in the middle state
		self.alpha = alpha
		self.beta = beta
		
		# Parameters for introducing detuning. Detuning percent is the percent of omega_max
		# If detuning_fixed = False, then at each timestep a value of detuning will bet 
		# Stochastically sampled from range dictated by detuning_percent and centered on 0.
		self.detuning = detuning
		self.detuning_percent = detuning_percent
		self.detuning_fixed = detuning_fixed
		
		# To introduce dephasing into the system. dephasing_sigma plays the same role as 
		# detuning_percent
		self.dephasing = dephasing
		self.dephasing_sigma = dephasing_sigma
		
		# Verbosity of the observation that is passed to the neural network as input.
		# if 0: only the diagonal elements of the density matrix; 1: full abs() of density
		# matrix, and; 2: Fully density matrix and previous pulse values.
		self.observation_verbosity = observation_verbosity
		
		# The length of the recall window that defines the input state if obs_verbosity=4
		self.recall_len = recall_len
		
		# Define the observation space based on the observation_verbosity
		if self.observation_verbosity == 0:
			self.observation_space = spaces.Box(low=0, high=1, shape=((num_levels),), dtype=np.float32)
		elif self.observation_verbosity == 1:
			self.observation_space = spaces.Box(low=0, high=1, shape=((num_levels + self.num_couplings),), dtype=np.float32)
		elif self.observation_verbosity == 2:
			self.observation_space = spaces.Box(low=0, high=1, shape=((num_levels**2),), dtype=np.float32)
		elif self.observation_verbosity == 3:
			self.observation_space = spaces.Box(low=0, high=1, shape=((num_levels**2 + self.num_couplings),), dtype=np.float32)
		elif self.observation_verbosity == 4:
			self.observation_space = spaces.Box(low=0, high=1, shape=(self.recall_len*(num_levels**2),), dtype=np.float32)
		
		
	def reset(self):
		"""
		Important: the observation must be a numpy array
		:return: (np.array) Initial observation to be usd in training. depends on obs_verbose
		"""
		ctap = CTAP(self.alpha, self.beta, self.omega_max, self.num_levels)
		# Initialize the agent at the right of the grid
		self.state, self.reduced_state_observation = ctap.initial_state()
		self.time_step = 0
		self.omegas = np.zeros(self.num_couplings)
		recall_array = np.zeros((self.recall_len*(self.num_levels**2),))
		recall_array[-(self.num_levels**2):] = np.absolute(self.state)[:]
		self.recall_deque = deque(recall_array)
		if self.observation_verbosity == 0:
			observation = self.reduced_state_observation
		elif self.observation_verbosity == 1:
			observation = np.concatenate((self.reduced_state_observation, self.omegas), axis=0)
		elif self.observation_verbosity == 2:
			observation = np.absolute(self.state)
		elif self.observation_verbosity == 3:
			observation = np.concatenate((np.absolute(self.state), self.omegas), axis=0)
		elif self.observation_verbosity == 4:
			observation = np.array(self.recall_deque)
			
		
			
		# here we convert to float32 to make it more general (in case we want to use continuous actions)
		return observation.astype(np.float32)
		

	def step(self, action):
	
		"""
		Defines 1 single step in the agent - environment interaction.
		
		:Args:
			action (np.array): An array containing the values of the control pulses to be used
							   in the next timestep.
		:Returns:
			observation (np.array): The updated observation after applying action in the environement
			reward (float): 	The reward obtained for the particular state, action, new state transition.
			done (bool): 	A boolean term that tells the agent whether the episode has come to an end
			info (dict): 	A dictionary to contain any other information one would like to extract
		"""
		ctap = CTAP(self.alpha, self.beta, self.omega_max, self.num_levels)
		
		self.get_action(action)
		
		state_observation, self.reduced_state_observation = ctap.evolve(self.state, self.omegas, self.detuning, 
																		self.detuning_percent, self.detuning_fixed,
															 			self.dephasing, self.dephasing_sigma, 
															 			self.timestep)
		
		reward = ctap.get_reward(self.reduced_state_observation)
		
		self.time_step +=1
		
		self.state = state_observation
		
		
		if self.observation_verbosity == 0:
			observation = self.reduced_state_observation
		elif self.observation_verbosity == 1:
			observation = np.concatenate((self.reduced_state_observation, self.omegas), axis=0)
		elif self.observation_verbosity == 2:
			observation = np.absolute(self.state)
		elif self.observation_verbosity == 3:
			observation = np.concatenate((np.absolute(self.state), self.omegas), axis=0)
		elif self.observation_verbosity == 4:
			for i in range(self.num_levels**2):
				self.recall_deque.popleft()
				self.recall_deque.append(np.absolute(state_observation)[i])
			observation = np.array(self.recall_deque)
			
		if self.num_levels == 3:
			done = bool(np.abs(self.reduced_state_observation[2]) > 0.995 or self.time_step == self.max_time)	  
		elif self.num_levels == 5:
			done = bool(np.abs(self.reduced_state_observation[4]) > 0.995 or self.time_step == self.max_time) 

		# Optionally we can pass additional info, we are not using that for now
		info = {}

		return observation.astype(np.float32), reward, done, info

	def render(self, mode='console'):
		pass
	def close(self):
		pass
	
	
"""
----------------------------------------------------------------------------------------------------------------------
"""

class StepEnv(ControlEnv, gym.Env):
	"""
	Custom Environment that follows gym interface.
	Environment that simulates the evolution of three inter-connected quantum dots
	with no detuning or loss.
	
	
	Action Space: Agent can take either a 2*positive, positive, negative, 2*negative
	or null step of a fixed	 step size.
	"""

	
	def __init__(self, max_step_height, step_size, num_levels=3, omega_max=100, omega_multiplier=[1,1], max_time=50, 
				timestep=0.025, alpha=1, beta=3, detuning=False, detuning_percent=0, 
				detuning_fixed=False, dephasing=False, dephasing_sigma=0,  
				observation_verbosity=2, recall_len=4):
				
		super(StepEnv, self).__init__(num_levels = num_levels, omega_max=omega_max, omega_multiplier=omega_multiplier, max_time=max_time, 
									timestep=timestep, alpha=alpha, beta=beta, detuning=detuning, 
									detuning_fixed=detuning_fixed, detuning_percent=detuning_percent, 
									dephasing=dephasing, dephasing_sigma=dephasing_sigma,  
									observation_verbosity=observation_verbosity, recall_len=recall_len)
		
		# The max number of fixed sized 'steps' the agent can take in on timestep
		self.max_step_height = max_step_height
		# Multiplier to define the height of each single step
		self.step_size = step_size
		
		# number of actions available for each pulse value. ranges from [-max_step_height, max_step_height]
		self.num_individual_actions = (2*self.max_step_height + 1)
		#Total number of actions 
		n_actions = self.num_individual_actions**self.num_couplings
		# whether to include detuning. (Bool)
		
		# Define action and observation space
		# They must be gym.spaces objects
		self.action_space = spaces.Discrete(n_actions)
		
	def get_action(self, action):
		"""
		Given a value in the action_space this function returns an np.array() of size 2 contaning the 
		corresponding pulse values.
		"""
		if self.num_levels == 3:
		
			self.omegas[0] = np.clip(self.omegas[0] + 
								 	((-self.max_step_height) + 
								  	(action//self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[0])
			self.omegas[1] = np.clip(self.omegas[1] + 
								 	((-self.max_step_height) + 
								 	 (action%self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[1])
		elif self.num_levels == 5:
			self.omegas[0] = np.clip(self.omegas[0] + 
								 	((-self.max_step_height) + 
								  	((action// self.num_individual_actions**2)//self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[0])
			self.omegas[1] = np.clip(self.omegas[1] + 
								 	((-self.max_step_height) + 
								 	 ((action// self.num_individual_actions**2)%self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[1])
			self.omegas[2] = np.clip(self.omegas[2] + 
								 	((-self.max_step_height) + 
								  	((action % self.num_individual_actions**2)//self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[2])
			self.omegas[3] = np.clip(self.omegas[3] + 
								 	((-self.max_step_height) + 
								 	 ((action% self.num_individual_actions**2)%self.num_individual_actions))*self.step_size, 0, self.omega_max*self.omega_multiplier[3])
		return self.omegas
	
	
"""
-------------------------------------------------------------------------------------------------------------------------------------
"""	  

class DiscreteEnv(ControlEnv, gym.Env):
	"""
	Custom Environment that follows gym interface.
	Environment that simulates the evolution of three inter-connected quantum dots
	with no detuning or loss.
	
	
	Action Space: Agent can take any one of a discrete set of values between 0 and 
	Omega max. The number of available values is given by a 'discreteness' value. i.e 
	Omega_max/(discreteness - 1). (The -1 since 0 values are also allowed).
	"""
	
	# Because of google colab, we cannot implement the GUI ('human' render mode)
	#metadata = {'render.modes': ['console']}
	def __init__(self, discreteness, num_levels=3, omega_max=100, omega_multiplier=[1,1], max_time=50, timestep = 0.025, 
				alpha=1, beta=3, detuning = False, detuning_percent=0.0, detuning_fixed = False, 
				dephasing = False, dephasing_sigma = 0.0, observation_verbosity = 2, recall_len=4):
				
		super(DiscreteEnv, self).__init__(num_levels = num_levels, omega_max=omega_max, omega_multiplier=omega_multiplier, 
											max_time=max_time, timestep=timestep, alpha=alpha, 
											beta=beta, detuning=detuning, detuning_fixed=detuning_fixed, 
											detuning_percent=detuning_percent, dephasing=dephasing, 
											dephasing_sigma=dephasing_sigma,  
											observation_verbosity= observation_verbosity,
											recall_len=recall_len)
		
		
		self.discreteness = discreteness
		
		# Define action and observation space
		# They must be gym.spaces objects
		n_actions = self.discreteness**self.num_couplings
		self.action_space = spaces.Discrete(n_actions)
		
		
		
		
	def get_action(self, action):
		"""
		Given a value in the action_space this function returns an np.array() of size 2 contaning the 
		corresponding pulse values. 
		
		Discretizes the range [0,OmegaMax] into 'discreteness' evenly spaced values.
		"""
		if self.num_levels == 3:
			self.omegas[0] = (action//self.discreteness)*(self.omega_max*self.omega_multiplier[0]/(self.discreteness -1))
			self.omegas[1] = (action% self.discreteness)*(self.omega_max*self.omega_multiplier[1]/(self.discreteness -1))
		elif self.num_levels == 5:
			self.omegas[0] = ((action // self.discreteness**2) // self.discreteness)*(self.omega_max*self.omega_multiplier[0]/(self.discreteness -1))
			self.omegas[1] = ((action // self.discreteness**2) % self.discreteness)*(self.omega_max*self.omega_multiplier[1]/(self.discreteness -1))
			self.omegas[2] = ((action % self.discreteness**2) // self.discreteness)*(self.omega_max*self.omega_multiplier[2]/(self.discreteness -1))
			self.omegas[3] = ((action % self.discreteness**2) % self.discreteness)*(self.omega_max*self.omega_multiplier[3]/(self.discreteness -1))
		return self.omegas
	
	
	
	
"""
-----------------------------------------------------------------------------------------
"""	   

	
class ContinuousEnv(ControlEnv, gym.Env):
	"""
	Custom Environment that follows gym interface.
	Environment that simulates the evolution of three inter-connected quantum dots
	with no detuning or loss.
	
	
	Action Space: Agent can take any value in the continuous interval [0, OmegaMax].
	***Suitable only for continuous action space algorithms like DDPG etc...
	"""
	

	def __init__(self,  num_levels, omega_max = 100, omega_multiplier=[1,1], max_time = 50, timestep = 0.025, alpha = 1, 
				beta = 3, detuning = False, detuning_percent =0, detuning_fixed = False, 
				dephasing = False, dephasing_sigma = 0, observation_verbosity = 2, recall_len=4):
				
		super(ContinuousEnv, self).__init__(num_levels = num_levels, omega_max=omega_max, omega_multiplier=omega_multiplier, 
											max_time=max_time, timestep=timestep, alpha=alpha, 
											beta=beta, detuning=detuning, detuning_fixed=detuning_fixed, 
											detuning_percent=detuning_percent, dephasing=dephasing, 
											dephasing_sigma=dephasing_sigma,  
											observation_verbosity= observation_verbosity,
											recall_len=recall_len)
		
		# Define action and observation space
		# They must be gym.spaces objects
		self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_couplings,), dtype = np.float32)
		
		
	def get_action(self, action):
		"""
		Given a value in the action_space this function returns an np.array() of size 2 contaning the 
		corresponding pulse values.
		"""
		self.omegas[0] = (action[0]+1)*(self.omega_max/2)*self.omega_multiplier[0]
		self.omegas[1] = (action[1]+1)*(self.omega_max/2)*self.omega_multiplier[1]
		return self.omegas
	
