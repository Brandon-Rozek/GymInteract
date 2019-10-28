import rltorch

sneaky_config = {}
sneaky_config['learning_rate'] = 1e-4
sneaky_config['target_sync_tau'] = 1e-3
sneaky_config['discount_rate'] = 0.99
sneaky_config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.1, iterations = 10**5)
# Number of episodes for the computer to train the agent without the human seeing
sneaky_config['replay_skip'] = 14
sneaky_config['batch_size'] = 32 * (sneaky_config['replay_skip'] + 1)
sneaky_config['memory_size'] = 10**4
