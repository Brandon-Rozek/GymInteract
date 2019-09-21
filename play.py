import gym
import pygame
import sys
import time
import matplotlib
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass


import pyglet.window as pw

from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread, Event, Timer

class Play:
    def __init__(self, env, action_selector, memory, agent, transpose = True, fps = 30, zoom = None, keys_to_action = None):
        self.env = env
        self.action_selector = action_selector
        self.transpose = transpose
        self.fps = fps
        self.zoom = zoom
        self.keys_to_action = None
        self.video_size = (0, 0)
        self.pressed_keys = []
        self.screen = None
        self.relevant_keys = set()
        self.running = True
        self.switch = Event()
        self.state = 0
        self.paused = False
        self.memory = memory
        self.agent = agent
        print("FPS ", 30)
    
    def _display_arr(self, obs, screen, arr, video_size):
        if obs is not None:
            if len(obs.shape) == 2:
                obs = obs[:, :, None]
            if obs.shape[2] == 1:
                obs = obs.repeat(3, axis=2)
            arr_min, arr_max = arr.min(), arr.max()
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
            pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if self.transpose else arr)
            pyg_img = pygame.transform.scale(pyg_img, video_size)
            screen.blit(pyg_img, (0,0))

    def _human_play(self, obs):
        action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
        prev_obs = obs
        obs, reward, env_done, _ = self.env.step(action)
        self._display_arr(obs, self.screen, self.env.unwrapped._get_obs(), video_size=self.video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in self.relevant_keys:
                    self.pressed_keys.append(event.key)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.KEYUP:
                if event.key in self.relevant_keys:
                    self.pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                self.video_size = event.size
                self.screen = pygame.display.set_mode(self.video_size)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return prev_obs, action, reward, obs, env_done
    
    def _computer_play(self, obs):
        prev_obs = obs
        action = self.action_selector.act(obs)
        obs, reward, env_done, _ = self.env.step(action)
        self._display_arr(obs, self.screen, self.env.unwrapped._get_obs(), video_size=self.video_size)

        # process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                self.video_size = event.size
                self.screen = pygame.display.set_mode(self.video_size)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        return prev_obs, action, reward, obs, env_done
    
    def _setup_video(self):
        if self.transpose:
            video_size = self.env.unwrapped.observation_space.shape[1], self.env.unwrapped.observation_space.shape[0]
        else:
            video_size = self.env.unwrapped.observation_space.shape[0], self.env.unwrapped.observation_space.shape[1]

        if self.zoom is not None:
            video_size = int(video_size[0] * self.zoom), int(video_size[1] * self.zoom)

        self.video_size = video_size
        self.screen = pygame.display.set_mode(self.video_size)
        pygame.font.init() # For later text
    
    def _setup_keys(self):
        if self.keys_to_action is None:
            if hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
                self.keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                assert False, self.env.spec.id + " does not have explicit key to action mapping, " + \
                            "please specify one manually"
        self.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
    
    def _increment_state(self):
        self.state = (self.state + 1) % 4

    def pause(self, text = ""):
        self.paused = True
        myfont = pygame.font.SysFont('Comic Sans MS', 50)
        textsurface = myfont.render(text, False, (0, 0, 0))
        self.screen.blit(textsurface,(0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                self.video_size = event.size
                self.screen = pygame.display.set_mode(self.video_size)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pressed_keys.append(event.key)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                self.pressed_keys.remove(event.key)
                self._increment_state()
                self.paused = False
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def start(self):
        """Allows one to play the game using keyboard.
        To simply play the game use:
            play(gym.make("Pong-v3"))
        Above code works also if env is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.
        If you wish to plot real time statistics as you play, you can use
        gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
        for last 5 second of gameplay.
            def callback(obs_t, obs_tp1, rew, done, info):
                return [rew,]
            env_plotter = EnvPlotter(callback, 30 * 5, ["reward"])
            env = gym.make("Pong-v3")
            play(env, callback=env_plotter.callback)
        Arguments
        ---------
        env: gym.Env
            Environment to use for playing.
        transpose: bool
            If True the output of observation is transposed.
            Defaults to true.
        fps: int
            Maximum number of steps of the environment to execute every second.
            Defaults to 30.
        zoom: float
            Make screen edge this many times bigger
        callback: lambda or None
            Callback if a callback is provided it will be executed after
            every step. It takes the following input:
                obs_t: observation before performing action
                obs_tp1: observation after performing action
                action: action that was executed
                rew: reward that was received
                done: whether the environment is done or not
                info: debug info
        keys_to_action: dict: tuple(int) -> int or None
            Mapping from keys pressed to action performed.
            For example if pressed 'w' and space at the same time is supposed
            to trigger action number 2 then key_to_action dict would look like this:
                {
                    # ...
                    sorted(ord('w'), ord(' ')) -> 2
                    # ...
                }
            If None, default key_to_action mapping for that env is used, if provided.
        """
        obs_s = self.env.unwrapped.observation_space
        assert type(obs_s) == gym.spaces.box.Box
        assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

        self._setup_keys()
        self._setup_video()
        
        self.clock = pygame.time.Clock()
        
        # States
        COMPUTER_PLAY = 0
        HUMAN_PLAY = 2

        env_done = True
        prev_obs = None
        obs = None
        reward = 0
        i = 0
        while self.running:
            if env_done:
                obs = self.env.reset()
                env_done = False
            
            if self.state == 0:
                prev_obs, action, reward, obs, env_done = self._computer_play(obs)
            elif self.state == 1:
                self.pause("Your Turn! Press <Space> to Start")
            elif self.state == 2:
                prev_obs, action, reward, obs, env_done = self._human_play(obs)
            elif self.state == 3:
                self.pause("Computers Turn! Press <Space> to Start")

            if self.state is COMPUTER_PLAY or self.state is HUMAN_PLAY:
                self.memory.append(prev_obs, action, reward, obs, env_done)
                
            if not self.paused:
                i += 1
                if i % (self.fps * 30) == 0: # Every 30 seconds...
                    print("TRAINING...")
                    self.agent.learn()
                    print("PAUSING...")
                    self._increment_state()
                    i = 0


        pygame.quit()

