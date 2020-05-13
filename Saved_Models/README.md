File Name:                                       Description:
------------                                     -------------
PPO1_ctap6.zip                                   A first attempt at using PPO for CTAP with no detuning:
                                                 Timesteps: 100000
                                                 entcoeff:  0.1
                                                 alpha:     1
                                                 beta:      3
                                                 
PPO1_Stirap_Detuned.zip                          A first attempt at using PPO for CTAP with detuning using StepEnv:
                                                 Step_height: 5
                                                 Timesteps: 300000
                                                 entcoeff:  0.1
                                                 alpha:     1
                                                 beta:      3
                                                 omega_max: 100
                                                 avg_max_time: 35
                                                 
DDQN_Stirap1.zip                                 A first attempt at DDQN with StepEnv:
                                                 env = StepEnv(max_time = 100, omega_max = 100, max_step_height = 2, step_size =                                                                5, detuning = False, alpha = 1, beta = 4)
                                                 model = DQN(deepq.MlpPolicy, env, verbose = 1, learning_rate = 1e-3,                                                                          exploration_fraction = 0.5, exploration_initial_eps = 1.0,                                                                        exploration_final_eps = 10.02, batch_size = 10, double_q = True,
                                                             prioritized_replay=True)
                                                 model.learn(total_timesteps = 50000)
                                                 
                                                 
DDQN_Stirap_Detuned1.zip                         env = StepEnv(max_time = 100, omega_max = 100, max_step_height = 2, 
                                                               step_size = 5, detuning = True, alpha = 1, beta = 4 )
                                                 model = DQN(deepq.MlpPolicy, env, verbose = 1, learning_rate = 1e-3,                                                                          exploration_fraction = 0.5, exploration_initial_eps = 1.0,                                                                        exploration_final_eps = 0.02, batch_size = 10, double_q = True,
                                                             prioritized_replay=True)
                                                 model.learn(total_timesteps = 500000)