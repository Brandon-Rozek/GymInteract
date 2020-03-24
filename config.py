import rltorch

config = {}
config['seed'] = 901
config['zoom'] = 4
config['environment_name'] = 'PongNoFrameskip-v4'
config['learning_rate'] = 1e-4
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.02, iterations = 10**5)
config['replay_skip'] = 4
config['batch_size'] = 32 * (config['replay_skip'] + 1)
config['num_sneaky_episodes'] = 10 # per loop
config['disable_cuda'] = False

config['seconds_play_per_state'] = 120
config['seconds_play_per_state'] = 5
# 30 transitions per second for 120 seconds = 3600 transitions per turn
config['memory_size'] = 86400
config['dqfd_demo_loss_weight'] = 0.01
config['dqfd_td_loss_weight'] = 1.
config['demo_prio_bonus'] = 0.
config['observed_prio_bonus'] = 0.

# Prioritized vs Random Sampling
# 0 - Random sampling
# 1 - Only the highest prioirities
config['prioritized_replay_sampling_priority'] = 0.6
config['prioritized_replay_sampling_priority'] = 0.
# How important are the weights for the loss?
# 0 - Treat all losses equally
# 1 - Lower the importance of high losses
# Should ideally start from 0 and move your way to 1 to prevent overfitting
config['prioritized_replay_weight_importance'] = rltorch.scheduler.ExponentialScheduler(initial_value = 0.4, end_value = 1, iterations = 10**5)
config['prioritized_replay_weight_importance'] = 0.
