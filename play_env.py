
# TODO: I'm kinda using this project to pilot the whole config/network/example separation
# The motivation behind this is that the file sizes are getting large and its increasing cognitive load :(

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

# Import my custom RL library
import rltorch
from rltorch.memory import PrioritizedReplayMemory, ReplayMemory, DQfDMemory
from rltorch.action_selector import EpsilonGreedySelector, ArgMaxSelector
import rltorch.env as E
import rltorch.network as rn

# Import OpenAI gym and related packages
from gym import make as makeEnv
from gym import Wrapper as GymWrapper
from gym.wrappers import Monitor as GymMonitor
import play


#
## Networks (Probably want to move this to config file)
#
from networks import Value

#
## Play Related Classes
#
class PlayClass(Thread):
  def __init__(self, env, action_selector, agent, sneaky_env, sneaky_actor, sneaky_agent, record_lock, config, sneaky_config):
    super(PlayClass, self).__init__()
    self.play = play.Play(env, action_selector, agent, sneaky_env, sneaky_actor, sneaky_agent, record_lock, config, sneaky_config)

  def run(self):
    self.play.start()

class Record(GymWrapper):
  def __init__(self, env, memory, lock, args):
    GymWrapper.__init__(self, env)
    self.memory = memory
    self.lock = lock # Lock for memory access
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
      self.lock.acquire()
      self.memory.append((state, action, reward, next_state, done))
      self.lock.release()
      self.current_i = 0
    return next_state, reward, done, info
  
  def log_transitions(self):
    if len(self.memory) > 0:
      basename = self.logdir + "/{}.{}".format(self.environment_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%s"))
      print("Base Filename: ", basename, flush = True)
      state, action, reward, next_state, done = zip(*self.memory)
      np_save(basename + "-state.npy", np_array(state), allow_pickle = False)
      np_save(basename + "-action.npy", np_array(action), allow_pickle = False)
      np_save(basename + "-reward.npy", np_array(reward), allow_pickle = False)
      np_save(basename + "-nextstate.npy", np_array(next_state), allow_pickle = False)
      np_save(basename + "-done.npy", np_array(done), allow_pickle = False)
      self.memory.clear()


## Parsing arguments
parser = ArgumentParser(description="Play and log the environment")
parser.add_argument("--environment_name", type=str, help="The environment name in OpenAI gym to play.")
parser.add_argument("--logdir", type=str, help="Directory to log video and (state, action, reward, next_state, done) in.")
parser.add_argument("--skip", type=int, help="Number of frames to skip logging.")
parser.add_argument("--fps", type=int, help="Number of frames per second")
parser.add_argument("--model", type=str, help = "The path location of the PyTorch model")
args = vars(parser.parse_args())

## Main configuration for script
from config import config
from sneaky_config import sneaky_config

# Environment name and log directory is vital so show help message and exit if not provided
if args['environment_name'] is None or args['logdir'] is None:
  parser.print_help()
  exit(1)

# Number of frames to skip when recording and fps can have sane defaults
if args['skip'] is None:
  args['skip'] = 3
if 'fps' not in args:
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
record_memory = []
record_lock = Lock()
env = Record(makeEnv(args['environment_name']), record_memory, record_lock, args)

# Bind record_env to current env so that we can reference log_transitions easier later
record_env = env

# Use native gym  monitor to get video recording
env = GymMonitor(env, args['logdir'], force=True)

# Preprocess enviornment
env = wrap_preprocessing(env)

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
memory =  DQfDMemory(capacity= config['memory_size'], alpha = config['prioritized_replay_sampling_priority'], max_demo = config['memory_size'] // 2)
actor = ArgMaxSelector(net, action_size, device = device)
agent = rltorch.agents.DQfDAgent(net, memory, config, target_net = target_net)

# Use a different environment for when the computer trains on the side so that the current game state isn't manipuated
# Also use MaxEnvSkip to speed up processing
sneaky_env = wrap_preprocessing(makeEnv(args['environment_name']), MaxAndSkipEnv = True)
sneaky_memory = ReplayMemory(capacity = sneaky_config['memory_size'])
sneaky_actor = EpsilonGreedySelector(net, action_size, device = device, epsilon = sneaky_config['exploration_rate'])

sneaky_agent = rltorch.agents.DQNAgent(net, sneaky_memory, sneaky_config, target_net = target_net)

# Pass all this information into the thread that will handle the game play and start
playThread = PlayClass(env, actor, agent, sneaky_env, sneaky_actor, sneaky_agent, record_lock, config, sneaky_config)
playThread.start()

# While the play thread is running, we'll periodically log transitions we've encountered
while playThread.is_alive():
  playThread.join(60) 
  record_lock.acquire()
  print("Logging....", end = " ")
  record_env.log_transitions()
  record_lock.release()

# Save what's remaining after process died
record_lock.acquire()
record_env.log_transitions()
record_lock.release()