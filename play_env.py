
# Import Python Standard Libraries
from threading import Thread, Lock
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime

# Import Pytorch related packages for NNs
from numpy import array as np_array
from numpy import save as np_save
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

# Import my custom RL library
import rltorch
from rltorch.memory import PrioritizedReplayMemory
from rltorch.action_selector import EpsilonGreedySelector
import rltorch.env as E
import rltorch.network as rn

# Import OpenAI gym and related packages
from gym import make as makeEnv
from gym import Wrapper as GymWrapper
from gym.wrappers import Monitor as GymMonitor
import play


#
## Networks
#
class Value(nn.Module):
  def __init__(self, state_size, action_size):
    super(Value, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    
    self.conv1 = nn.Conv2d(4, 32, kernel_size = (8, 8), stride = (4, 4))
    self.conv2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride = (2, 2))    
    self.conv3 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1))
    
    self.fc1 = nn.Linear(3136, 512)
    self.fc1_norm = nn.LayerNorm(512)

    self.value_fc = rn.NoisyLinear(512, 512)
    self.value_fc_norm = nn.LayerNorm(512)
    self.value = nn.Linear(512, 1)
    
    self.advantage_fc = rn.NoisyLinear(512, 512)
    self.advantage_fc_norm = nn.LayerNorm(512)
    self.advantage = nn.Linear(512, action_size)

  
  def forward(self, x):
    x = x.float() / 256
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    
    # Makes batch_size dimension again
    x = x.view(-1, 3136)
    x = F.relu(self.fc1_norm(self.fc1(x)))
    
    state_value = F.relu(self.value_fc_norm(self.value_fc(x)))
    state_value = self.value(state_value)
    
    advantage = F.relu(self.advantage_fc_norm(self.advantage_fc(x)))
    advantage = self.advantage(advantage)
    
    x = state_value + advantage - advantage.mean()
    
    # For debugging purposes...
    if torch.isnan(x).any().item():
      print("WARNING NAN IN MODEL DETECTED")
    
    return x


#
## Play Related Classes
#
Transition = namedtuple('Transition',
      ('state', 'action', 'reward', 'next_state', 'done'))

class PlayClass(Thread):
  def __init__(self, env, action_selector, memory, memory_lock, agent, sneaky_env, config):
    super(PlayClass, self).__init__()
    self.play = play.Play(env, action_selector, memory, memory_lock, agent, sneaky_env, config)

  def run(self):
    self.play.start()

class Record(GymWrapper):
  def __init__(self, env, memory, memory_lock, args):
    GymWrapper.__init__(self, env)
    self.memory_lock = memory_lock
    self.memory = memory
    self.skipframes = args['skip']
    self.environment_name = args['environment_name']
    self.logdir = args['logdir']
    self.current_i = 0

  def reset(self):
    return self.env.reset()

  def step(self, action):
    state = self.env.env._get_obs()
    next_state, reward, done, info = self.env.step(action)
    self.current_i += 1
    # Don't add to memory until a certain number of frames is reached
    if self.current_i % self.skipframes == 0:
      self.memory_lock.acquire()
      self.memory.append(state, action, reward, next_state, done)
      self.memory_lock.release()
      self.current_i = 0
    return next_state, reward, done, info
  
  def log_transitions(self):
    self.memory_lock.acquire()
    if len(self.memory) > 0:
      basename = self.logdir + "/{}.{}".format(self.environment_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%s"))
      print("Base Filename: ", basename)
      state, action, reward, next_state, done = zip(*self.memory)
      np_save(basename + "-state.npy", np_array(state), allow_pickle = False)
      np_save(basename + "-action.npy", np_array(action), allow_pickle = False)
      np_save(basename + "-reward.npy", np_array(reward), allow_pickle = False)
      np_save(basename + "-nextstate.npy", np_array(next_state), allow_pickle = False)
      np_save(basename + "-done.npy", np_array(done), allow_pickle = False)
      self.memory.clear()
    self.memory_lock.release()


## Parsing arguments
parser = ArgumentParser(description="Play and log the environment")
parser.add_argument("--environment_name", type=str, help="The environment name in OpenAI gym to play.")
parser.add_argument("--logdir", type=str, help="Directory to log video and (state, action, reward, next_state, done) in.")
parser.add_argument("--skip", type=int, help="Number of frames to skip logging.")
parser.add_argument("--fps", type=int, help="Number of frames per second")
parser.add_argument("--model", type=str, help = "The path location of the PyTorch model")
args = vars(parser.parse_args())

## Main configuration for script
config = {}
config['seed'] = 901
config['seconds_play_per_state'] = 60
config['zoom'] = 4
config['environment_name'] = 'PongNoFrameskip-v4'
config['learning_rate'] = 1e-4
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.1, iterations = 10**5)
# Number of episodes for the computer to train the agent without the human seeing
config['num_sneaky_episodes'] = 20
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


# Environment name and log directory is vital so show help message and exit if not provided
if args['environment_name'] is None or args['logdir'] is None:
  parser.print_help()
  exit(1)

# Number of frames to skip when recording and fps can have sane defaults
if args['skip'] is None:
  args['skip'] = 3
if args['fps'] is None:
  args['fps'] = 30


def wrap_preprocessing(env, MaxAndSkipEnv = False):
  env = E.NoopResetEnv(
          E.EpisodicLifeEnv(env),
          noop_max = 30
        )
  if MaxAndSkipEnv:
    env = E.MaxAndSkipEnv(env, skip = 4)
  return E.ClippedRewardsWrapper(
    E.FrameStack(
      E.TorchWrap(
        E.ProcessFrame84(
          E.FireResetEnv(env)
        )
      )
    , 4)
  )


## Set up environment to be recorded and preprocessed
memory = PrioritizedReplayMemory(capacity = config['memory_size'], alpha = config['prioritized_replay_sampling_priority'])
memory_lock = Lock()
env = Record(makeEnv(args['environment_name']), memory, memory_lock, args)
# Bind record_env to current env so that we can reference log_transitions easier later
record_env = env
# Use native gym  monitor to get video recording
env = GymMonitor(env, args['logdir'], force=True)
# Preprocess enviornment
env = wrap_preprocessing(env)

# Use a different environment for when the computer trains on the side so that the current game state isn't manipuated
# Also use MaxEnvSkip to speed up processing
sneaky_env = wrap_preprocessing(makeEnv(args['environment_name']), MaxAndSkipEnv = True)

# Set seeds
rltorch.set_seed(config['seed'])
env.seed(config['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Set up the networks
net = rn.Network(Value(state_size, action_size), 
                      Adam, config, device = device)
target_net = rn.TargetNetwork(net, device = device)

# Relevant components from RLTorch
actor = EpsilonGreedySelector(net, action_size, device = device, epsilon = config['exploration_rate'])
agent = rltorch.agents.DQNAgent(net, memory, config, target_net = target_net)

# Pass all this information into the thread that will handle the game play and start
playThread = PlayClass(env, actor, memory, memory_lock, agent, sneaky_env, config)
playThread.start()

# While the play thread is running, we'll periodically log transitions we've encountered
while playThread.is_alive():
  playThread.join(60) 
  print("Logging....", end = " ")
  record_env.log_transitions()

# Save what's remaining after process died
record_env.log_transitions()
