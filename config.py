import rltorch

config = {}
config['seed'] = 901
config['seconds_play_per_state'] = 120
config['zoom'] = 4
config['environment_name'] = 'PongNoFrameskip-v4'
config['learning_rate'] = 1e-4
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.1, iterations = 10**5)
# Number of episodes for the computer to train the agent without the human seeing
config['num_sneaky_episodes'] = 10
config['num_train_per_demo'] = 50 # 100 total since you have two demo training per cycle
config['replay_skip'] = 14
config['batch_size'] = 32 * (config['replay_skip'] + 1)
config['disable_cuda'] = False
config['memory_size'] = 10**4
# Prioritized vs Random Sampling
# 0 - Random sampling
# 1 - Only the highest prioirities
config['prioritized_replay_sampling_priority'] = 0.6
# How important are the weights for the loss?
# 0 - Treat all losses equally
# 1 - Lower the importance of high losses
# Should ideally start from 0 and move your way to 1 to prevent overfitting
config['prioritized_replay_weight_importance'] = rltorch.scheduler.ExponentialScheduler(initial_value = 0.4, end_value = 1, iterations = 10**5)
