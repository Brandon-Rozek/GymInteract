import rltorch

config = {}
config['seed'] = 901
config['zoom'] = 4
config['environment_name'] = 'PongNoFrameskip-v4'
config['learning_rate'] = 1e-5
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['disable_cuda'] = False

config['seconds_play_per_state'] = 120
# 30 transitions per second for 120 seconds = 3600 transitions per turn
config['memory_size'] = 21600 # To hold 6 demonstrations
config['batch_size'] = 64
config['num_train_per_demo'] = 115 # 4 looks * transitions per turn / (2 * batch_size)


# Prioritized vs Random Sampling
# 0 - Random sampling
# 1 - Only the highest prioirities
config['prioritized_replay_sampling_priority'] = 0.6
# How important are the weights for the loss?
# 0 - Treat all losses equally
# 1 - Lower the importance of high losses
# Should ideally start from 0 and move your way to 1 to prevent overfitting
config['prioritized_replay_weight_importance'] = rltorch.scheduler.ExponentialScheduler(initial_value = 0.4, end_value = 1, iterations = 10**5)
