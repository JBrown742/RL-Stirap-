#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from stable_baselines.common.callbacks import BaseCallback
import os

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
from matplotlib import gridspec

class CtapPlot(object):
	def __init__(self):
		pass
		
		
	def pop_pulse_plot(self,num_timesteps, model, env, deterministic=True):
		times = [n*env.timestep for n in range(num_timesteps)]
		#(env.omega_max*env.timestep)/np.pi)
		obs_vec = np.zeros((num_timesteps, env.num_levels**2), dtype = object)
		omegas_vec = np.zeros((num_timesteps, env.num_couplings))
		obs = env.reset()
		for i in range(num_timesteps):
			omegas_vec[i,:] = env.omegas[:]
			obs_vec[i,:] = np.abs(env.state[:])
			action = model.predict(obs, deterministic=deterministic)
			#print(action[0])
			obs, reward, done, info = env.step(action[0])

		plt.figure(1, figsize=(6,5))
		for i in range(env.num_levels):
			plt.step(times, obs_vec[:,(i*(env.num_levels+1))], label = 'rho'+str(i))
			plt.legend()
			plt.xlabel("t")
			plt.ylabel('populations')

		plt.figure(2, figsize=(6,5))
		for i in range(env.num_couplings):
			plt.step(times, omegas_vec[:,i], label = str(i))
			plt.legend()
			plt.xlabel("t")
		return
	
	def average_plots(self, num_timesteps, model, env, runs, deterministic = True):
		# Number of couplings, in the case of 3 dot this is 2
		# So the observation vector will have an array of size num_timesteps=40, each element and array of size 3 and
		# each element of the inner array contains runs elements corresponding to the rho values at that timestep for the
		# j runs.
		times = [n*env.timestep for n in range(num_timesteps)]
		#(env.omega_max*env.timestep)/np.pi)*
		obs_vec = np.zeros((num_timesteps, env.num_levels, runs), dtype = object)
		omegas_vec = np.zeros((num_timesteps, env.num_couplings, runs))
		
		for j in range(runs):
			obs = env.reset()
			for i in range(num_timesteps):
				omegas_vec[i,:,j] = env.omegas[:]
				obs_vec[i,:,j] = env.reduced_state_observation[:]
				#print(obs)
				action = model.predict(obs, deterministic = deterministic)
				
				obs, reward, done, info = env.step(action[0])


		average_obs_vec = np.zeros((num_timesteps, env.num_levels))
		average_omegas_vec = np.zeros((num_timesteps, env.num_levels))
		for k in range(num_timesteps):
			for t in range(3):
				average_obs_vec[k,t] = np.mean(obs_vec[k,t,:])
			for l in range(env.num_couplings):
				average_omegas_vec[k,l] = np.mean(omegas_vec[k,l,:])
		
		print(" Max average final population: {} \n Max final population: {} \n Max Average Population of 2: {} \n Max population of 2: {}".format(np.max(average_obs_vec[:,2]), np.max(obs_vec[:,2,:]), np.max(average_obs_vec[:,1]), np.max(obs_vec[:,1,:])))
		plt.figure(figsize=(10,4))
		plt.subplot(1,2,1)
		for i in range(env.num_levels):
			plt.plot(times, average_obs_vec[:,i], label = 'rho'+str(i))
		plt.legend()
		plt.xlabel("t")
		plt.ylabel("populations")
		plt.title("Population dynamics")
			
		plt.subplot(1,2,2)
		for i in range(env.num_couplings):
			plt.step(times, average_omegas_vec[:,i], label = str(i))
		plt.legend()
		plt.xlabel("t")
		plt.ylabel('Pulses')
		plt.title("Averaged Pulse patterns")
		plt.tight_layout()
		return
		
	def discrete_plots(self,num_timesteps, model, env, deterministic=True):
		times = [n*env.timestep for n in range(num_timesteps)]
		# (env.omega_max*env.timestep)/np.pi)*
		obs_vec = np.zeros((num_timesteps, env.num_levels), dtype = object)
		omegas_vec = np.zeros((num_timesteps, env.num_couplings))
		obs = env.reset()
		for i in range(num_timesteps):
			omegas_vec[i,:] = env.omegas[:]
			obs_vec[i,:] = np.abs(env.reduced_state_observation[:])
			action = model.predict(obs, deterministic=deterministic)
			#print(action[0])
			obs, reward, done, info = env.step(action[0])
				
		plt.figure(figsize=(5,7))
		gs = gridspec.GridSpec(3,1, height_ratios=[1, 3, 1], hspace=0.01) 
		plt.subplot(gs[0])
		plt.step(times, omegas_vec[:,0]/env.omega_max, label = str(0))
		plt.legend()
		plt.ylabel("Omega/Omega max")
				
		plt.subplot(gs[1])
		for i in range(env.num_levels):
			plt.step(times, obs_vec[:,i], label = 'rho'+str(i))
		plt.legend()	
		plt.ylabel('populations')
		plt.subplot(gs[2])
		plt.step(times, omegas_vec[:,1]/env.omega_max, label = str(1), color='orange')
		plt.legend()
		plt.xlabel("t")
		plt.ylabel("Omega/Omega max")
		
		print("Maximum final population: {}  \nMax middle population: {} \n Mean middle population: {}".format(str(np.max(obs_vec[:,2])), str(np.max(obs_vec[:,1])), str(np.mean(obs_vec[:,1]))))
			
		return
		
class PlottingCallback(BaseCallback):
	"""
	Callback for plotting the performance in realtime.

	:param verbose: (int)
	"""
	def __init__(self, log_dir, verbose=1):
		super(PlottingCallback, self).__init__(log_dir, verbose)
		self._plot = None
		self.log_dir=log_dir
	def _on_step(self) -> bool:
		# get the monitor's data
		x, y = ts2xy(load_results(self.log_dir), 'timesteps')
		if self._plot is None: # make the plot
			plt.ion()
			fig = plt.figure(figsize=(6,3))
			ax = fig.add_subplot(111)
			line, = ax.plot(x, y)
			self._plot = (line, ax, fig)
			plt.show()
		else: # update and rescale the plot
			self._plot[0].set_data(x, y)
			self._plot[-2].relim()
			self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02, 
								   self.locals["total_timesteps"] * 1.02])
			self._plot[-2].autoscale_view(True,True,True)
			self._plot[-1].canvas.draw()