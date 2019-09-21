import play
import rltorch
import rltorch.memory as M
import torch
import gym
from collections import namedtuple
from datetime import datetime
from rltorch.action_selector import EpsilonGreedySelector
import rltorch.env as E
import rltorch.network as rn
import torch.nn as nn
import torch.nn.functional as F
import pickle
import threading
from time import sleep
import argparse
import sys
import numpy as np


## CURRRENT ISSUE: MaxSkipEnv applies to the human player as well, which makes for an awkward gaming experience
# What are your thoughts? Training is different if expert isn't forced with the same constraint
# At some point I need to introduce learning

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



Transition = namedtuple('Transition',
      ('state', 'action', 'reward', 'next_state', 'done'))

class PlayClass(threading.Thread):
  def __init__(self, env, action_selector, memory, agent, fps = 60):
    super(PlayClass, self).__init__()
    self.env = env
    self.fps = fps
    self.play = play.Play(self.env, action_selector, memory, agent, fps = fps, zoom = 4)

  def run(self):
    self.play.start()

class Record(gym.Wrapper):
  def __init__(self, env, memory, args, skipframes = 3):
    gym.Wrapper.__init__(self, env)
    self.memory_lock = threading.Lock()
    self.memory = memory
    self.args = args
    self.skipframes = skipframes
    self.current_i = skipframes

  def reset(self):
    return self.env.reset()

  def step(self, action):
    self.memory_lock.acquire()
    state = self.env.env._get_obs()
    next_state, reward, done, info = self.env.step(action)
    if self.current_i <= 0:
      self.memory.append(Transition(state, action, reward, next_state, done))
      self.current_i = self.skipframes
    else: self.current_i -= 1
    self.memory_lock.release()
    return next_state, reward, done, info
  
  def log_transitions(self):
    self.memory_lock.acquire()
    if len(self.memory) > 0:
      basename = self.args['logdir'] + "/{}.{}".format(self.args['environment_name'], datetime.now().strftime("%Y-%m-%d-%H-%M-%s"))
      print("Base Filename: ", basename)
      state, action, reward, next_state, done = zip(*self.memory)
      np.save(basename + "-state.npy", np.array(state), allow_pickle = False)
      np.save(basename + "-action.npy", np.array(action), allow_pickle = False)
      np.save(basename + "-reward.npy", np.array(reward), allow_pickle = False)
      np.save(basename + "-nextstate.npy", np.array(next_state), allow_pickle = False)
      np.save(basename + "-done.npy", np.array(done), allow_pickle = False)
      self.memory.clear()
    self.memory_lock.release()


## Parsing arguments
parser = argparse.ArgumentParser(description="Play and log the environment")
parser.add_argument("--environment_name", type=str, help="The environment name in OpenAI gym to play.")
parser.add_argument("--logdir", type=str, help="Directory to log video and (state, action, reward, next_state, done) in.")
parser.add_argument("--skip", type=int, help="Number of frames to skip logging.")
parser.add_argument("--fps", type=int, help="Number of frames per second")
parser.add_argument("--model", type=str, help = "The path location of the PyTorch model")
args = vars(parser.parse_args())

config = {}
config['seed'] = 901
config['environment_name'] = 'PongNoFrameskip-v4'
config['learning_rate'] = 1e-4
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 1, end_value = 0.1, iterations = 10**5)
config['batch_size'] = 480
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



if args['environment_name'] is None or args['logdir'] is None:
  parser.print_help()
  sys.exit(1)

if args['skip'] is None:
  args['skip'] = 3

if args['fps'] is None:
  args['fps'] = 30

## Starting the game
memory = []
env = Record(gym.make(args['environment_name']), memory, args, skipframes = args['skip'])
record_env = env
env = gym.wrappers.Monitor(env, args['logdir'], force=True)
env = E.ClippedRewardsWrapper(
    E.FrameStack(
      E.TorchWrap(
        E.ProcessFrame84(
          E.FireResetEnv(
            # E.MaxAndSkipEnv(
              E.NoopResetEnv(
                E.EpisodicLifeEnv(gym.make(config['environment_name']))
              , noop_max = 30)
            # , skip=4)
          )
        )
      ),
    4)
  )

rltorch.set_seed(config['seed'])

device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

net = rn.Network(Value(state_size, action_size), 
                      torch.optim.Adam, config, device = device)
target_net = rn.TargetNetwork(net, device = device)

actor = EpsilonGreedySelector(net, action_size, device = device, epsilon = config['exploration_rate'])
memory = M.PrioritizedReplayMemory(capacity = config['memory_size'], alpha = config['prioritized_replay_sampling_priority'])
agent = rltorch.agents.DQNAgent(net, memory, config, target_net = target_net)

env.seed(config['seed'])

playThread = PlayClass(env, actor, memory, agent, args['fps'])
playThread.start()

## Logging portion
while playThread.is_alive():
  playThread.join(60) 
  print("Logging....", end = " ")
  record_env.log_transitions()

# Save what's remaining after process died
record_env.log_transitions()
