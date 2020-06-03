#!/usr/bin/env python
# coding: utf-8

# 

# In[30]:


# Import ctap_evolution.py


import numpy as np
import qutip as qt
import scipy as sci

class CTAP(object):
	def __init__(self,alpha,beta, omega_max, num_levels):
		self.alpha = alpha
		self.beta = beta
		self.num_levels = num_levels
		self.omega_max = omega_max
		self.num_couplings = self.num_levels - 1
		# defines a function that generates an initial density matrix of dimension NxN with an excitation 
		# in the e^th eigenstate

	def initial_state(self):
		"""
		Returns an np.array() object representing the initial state of the system with a
		single excitation in the first state. (rho_11 = 1).
		"""
		x=[]
		for i in range(self.num_levels**2):
			if i != 0:
				x.append(0.)
			else:
				x.append(1.)
		y = []
		for i in range(self.num_levels):
			if i != 0:
				y.append(0.)
			else:
				y.append(1.)
		return np.array(x), np.array(y)
	
	
	# Initialise state to be passed into the QuTIP Solver
	

	def get_reward(self, reduced_rho_out):
		"""
		Calculates the reward. Compatible with 
		all agents using SB. Follows the form in (arXiv:1901.06603v1 [quant-ph] 20 Jan 2019)
		:params:
		--------
		rho_out: The state of the system immediately following to the timestep.
		
		:returns:
		---------
		reward: the reward for the single agent-env interaction.
		"""
		
		if len(reduced_rho_out) == 3:
			out11 = np.abs(reduced_rho_out[0])
			out22 = np.abs(reduced_rho_out[1])
			out33 = np.abs(reduced_rho_out[2])
			
			reward= self.alpha * (out33 -1 -out22)
			if out22 > 0.005:
 				reward -= np.exp(3*self.beta*out22)
			if out33>0.995:
				reward+=50

		elif len(reduced_rho_out) == 5:
			reward = np.exp(self.beta*reduced_rho_out[4])
			if reduced_rho_out[4] > 0.99:
				reward +=100
		return reward

	def get_reward_VPG(self, rho_in, rho_out):
		"""
		Calculates the reward for 3 dot case after one agent-environment interaction. Compatible with the
		agents based on Marquardt's implementation of REINFORCE using keras.
		:params:
		--------
		rho_out: The state of the system immediately following to the timestep.
		rho_in: The state of the system immediately preceding the timestep.
		:returns:
		---------
		reward: the reward for the single agent-env interaction.
		"""
		in33 = np.abs(rho_in[8])
		in22 = np.abs(rho_in[4])
		in11 = np.abs(rho_in[0])
		out11 = np.abs(rho_out[0])
		out22 = np.abs(rho_out[4])
		out33 = np.abs(rho_out[8])
		reward = 0
		return reward

	



	def evolve(self, rho, omegas, detuning, detuning_percent, detuning_fixed, dephasing, dephasing_sigma, timestep):
		"""
		Constant state evolution of the system, for a single timestep, using the STIRAP 
		hamiltonian as in arXiv:1901.06603v1.
		:params:
		--------
		rho: (np.array) Input state of the systems (from either the previous timestep or the initial
						state)
		omegas: (np.array) An array of size 2 containing the constant pulse values for the pump and stokes
						   pulse respectively.
		omega_max: (float) Maximum value for the pulses
		detuning: (bool) True for inclusion of random detuning in the interval [-detuning_percent,detuning_percent]*omega_max
		detuning_percent: (float) Value between 0 and 1 defining the interval from which detuning is randomly sampled.
		:returns:
		---------
		rho_out: (np.array) State of the system after the evolution.
		reduced_rho_out: (np.array) Array containing only the diagonal elements of rho_out.
		"""
		tlist = [0.0, timestep]
		
		# Turn the listform of rho into a proper qobj for the solver
		rho_proper = qt.Qobj(rho.reshape(self.num_levels,self.num_levels))
		
		# Initialize an empty array to fill with the pulse values to make the hamiltonian.
		ham_array = np.zeros((self.num_levels, self.num_levels))
		
		for i in range(self.num_levels):
			for j in range(self.num_levels):
				if i == j+1:
					ham_array[i,j] = 0.5*omegas[j]
				elif j== i+1:
					ham_array[i,j] = 0.5*omegas[i]
		hamiltonian = qt.Qobj(ham_array)

		# If the detuning kwarg = True we introduce uniformly random detuning into
		# each timestep. 
		if detuning_fixed == False:
			delta1 = np.random.uniform(low = -detuning_percent*self.omega_max, high= detuning_percent*self.omega_max)
			delta2	= 0
		elif detuning_fixed == True:
			delta2 = detuning_percent*self.omega_max
			delta1 = detuning_percent*self.omega_max
		elif detuning_fixed == "pierpaolo":
			delta2 = detuning_percent*self.omega_max
			delta1 = -23.5*delta2
		
		detuning_hamiltonian = qt.Qobj([[0,0,0],
										[0,delta1,0],
										[0,0,delta2]
										])
		
		
		
		# Currently only compatablibe with num_levels = 3
		if detuning == True:
			H = hamiltonian + detuning_hamiltonian
			#print(H)
		else:
			H = hamiltonian
		#print(H)	
		# Dephasing section:
		Sigma = dephasing_sigma*self.omega_max
		c0 =  np.sqrt(Sigma)*qt.fock_dm(3,0)
		c1 =  np.sqrt(Sigma)*qt.fock_dm(3,1)
		c2 =  np.sqrt(Sigma)*qt.fock_dm(3,2)
		collapse_operators =[c0, c1, c2]
		# Pass the (stepwise constant) hamiltonian into the mesolve function to evolve the
		# state and access the resultant state using the .states attribute.
		if dephasing == True:
			rho_out = qt.mesolve(H,rho_proper, tlist, c_ops = collapse_operators)
		elif dephasing ==False:
			rho_out = qt.mesolve(H,rho_proper, tlist)
			
		x = rho_out.states[-1]
		
		# flatten the resultant state back into an array again. Also return a reduced_rho
		# containing only the diagonal elements of the density matrix.
		rho_final = x.full().reshape((self.num_levels**2,))
		
		reduced_rho_final = x.diag() 
	
		return rho_final, reduced_rho_final
		
		

# In[ ]:




