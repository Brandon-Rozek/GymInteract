import rltorch

sneaky_config = {}
sneaky_config['learning_rate'] = 1e-5
sneaky_config['target_sync_tau'] = 1e-3
sneaky_config['discount_rate'] = 0.99
sneaky_config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.02, iterations = 10**5)
# Number of episodes for the computer to train the agent without the human seeing
sneaky_config['replay_skip'] = 29 # Gradient descent every second
sneaky_config['batch_size'] = 16 * (sneaky_config['replay_skip'] + 1) # Calculated based on memory constraints
sneaky_config['memory_size'] = 2000 # batch_size * 2 looks = 66 seconds of gameplay
# Number of episodes for the computer to train the agent without the human seeing
sneaky_config['num_sneaky_episodes'] = 10