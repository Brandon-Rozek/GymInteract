from gym.spaces.box import Box
import pygame
from pygame.locals import VIDEORESIZE
from rltorch.memory import ReplayMemory

class Play:
    def __init__(self, env, action_selector, agent, sneaky_env, sneaky_actor, sneaky_agent, record_lock, config, sneaky_config):
        self.env = env
        self.action_selector = action_selector
        self.record_lock = record_lock
        self.record_locked = False
        self.sneaky_agent = sneaky_agent
        self.agent = agent
        self.sneaky_env = sneaky_env
        self.sneaky_actor = sneaky_actor
        # Get relevant parameters from config or set sane defaults
        self.transpose = config['transpose'] if 'transpose' in config else True
        self.fps = config['fps'] if 'fps' in config else 30
        self.zoom = config['zoom'] if 'zoom' in config else 1
        self.keys_to_action = config['keys_to_action'] if 'keys_to_action' in config else None
        self.seconds_play_per_state = config['seconds_play_per_state'] if 'seconds_play_per_state' in config else 30
        self.num_sneaky_episodes = sneaky_config['num_sneaky_episodes'] if 'num_sneaky_episodes' in sneaky_config else 10
        self.replay_skip = sneaky_config['replay_skip'] if 'replay_skip' in sneaky_config else 0
        self.num_train_per_demo = config['num_train_per_demo'] if 'num_train_per_demo' in config else 1
        # Initial values...
        self.video_size = (0, 0)
        self.pressed_keys = []
        self.screen = None
        self.relevant_keys = set()
        self.running = True
        self.state = 0
        self.clock = pygame.time.Clock()
        self.sneaky_iteration = 0
        self.paused = False
    
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
    
    def _process_common_pygame_events(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == VIDEORESIZE:
            self.video_size = event.size
            self.screen = pygame.display.set_mode(self.video_size)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
        elif not self.paused and self.state in [0, 3] and event.type == pygame.KEYUP and event.key == pygame.K_F1:
            self.paused = True
            self.display_text("Paused... Press F1 to unpause.")
        else:
            # No event was matched here
            return False
        # One of the events above matched
        return True
    

    def _human_play(self, obs):
        action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
        prev_obs = obs
        obs, reward, env_done, _ = self.env.step(action)
        self._display_arr(obs, self.screen, self.env.unwrapped._get_obs(), video_size=self.video_size)

        # process pygame events
        for event in pygame.event.get():
            if self._process_common_pygame_events(event):
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key in self.relevant_keys:
                    self.pressed_keys.append(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in self.relevant_keys:
                    if event.key in self.pressed_keys: # To make sure that program doesn't crash
                        self.pressed_keys.remove(event.key)

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
            self._process_common_pygame_events(event)
        
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
        pygame.font.init()
    
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
        self.state = (self.state + 1) % 5

    def transition(self, text = ""):
        myfont = pygame.font.SysFont('Comic Sans MS', 50)
        textsurface = myfont.render(text, False, (0, 0, 0))
        self.screen.blit(textsurface,(0,0))

        # Process pygame events
        for event in pygame.event.get():
            if self._process_common_pygame_events(event):
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pressed_keys.append(event.key)
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                self.pressed_keys.remove(event.key)
                self._increment_state()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def sneaky_train(self):
        # self.record_lock.acquire()
        # Do a standard RL algorithm process for a certain number of episodes
        for i in range(self.num_sneaky_episodes):
            print("Episode: %d / %d, Reward: " % ((self.num_sneaky_episodes * self.sneaky_iteration) + i + 1, (self.sneaky_iteration + 1) * self.num_sneaky_episodes), end = "")

            # Reset all episode related variables
            prev_obs = self.sneaky_env.reset()
            done = False
            step = 0
            total_reward = 0
            
            while not done:
                action = self.sneaky_actor.act(prev_obs)
                obs, reward, done, _ = self.sneaky_env.step(action)
                total_reward += reward
                self.sneaky_agent.memory.append(prev_obs, action, reward, obs, done)
                prev_obs = obs
                step += 1
                if step % self.replay_skip == 0:
                    self.sneaky_agent.learn()
            
            # Finish the previous print with the total reward obtained during the episode
            print(total_reward, flush = True)
        self.sneaky_iteration += 1
        # self.record_lock.release()
    
    def display_text(self, text):
        myfont = pygame.font.SysFont('Comic Sans MS', 50)
        textsurface = myfont.render(text, False, (0, 0, 0))
        self.screen.blit(textsurface,(0,0))
        pygame.display.flip()
    
    def clear_text(self, obs):
        self._display_arr(obs, self.screen, self.env.unwrapped._get_obs(), video_size=self.video_size)
        pygame.display.flip()
    
    def process_pause_state(self, obs):
        # Process game events
        for event in pygame.event.get():
            # This rule needs to be before the common one otherwise unpausing is ignored
            if event.type == pygame.KEYUP and event.key == pygame.K_F1:
                self.paused = False
                self.clear_text(obs)
                if self.record_locked:
                    self.record_lock.release()
                    self.record_locked = False
            else:
                self._process_common_pygame_events(event)

    def start(self):
        """Allows one to play the game using keyboard.
        To simply play the game use:
            play(gym.make("Pong-v3"))
        Above code works also if env is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.
        """
        obs_s = self.env.unwrapped.observation_space
        assert type(obs_s) == Box
        assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

        self._setup_keys()
        self._setup_video()
        
        # States
        HUMAN_PLAY = 0
        SNEAKY_COMPUTER_PLAY = 1
        TRANSITION = 2
        COMPUTER_PLAY = 3
        TRANSITION2 = 4
        
        env_done = True
        prev_obs = None
        action = None
        reward = None
        obs = None
        i = 0
        episode_num = 0
        while self.running:
            # If the environment is done after a turn, reset it so we can keep going
            if env_done:
                episode_num += 1
                print("Human/Computer Episode:", episode_num, flush = True)
                obs = self.env.reset()
                env_done = False
            
            if self.paused:
                if not self.record_locked:
                    self.record_lock.acquire()
                    self.record_locked = True
                self.process_pause_state(obs)
                continue

            if self.state is HUMAN_PLAY:
                if self.record_locked:
                    self.record_lock.release()
                    self.record_locked = False
                prev_obs, action, reward, obs, env_done = self._human_play(obs)
            
            # The computer will train for a few episodes without showing to the user.
            # Mainly to speed up the learning process a bit
            elif self.state is SNEAKY_COMPUTER_PLAY:
                if not self.record_locked:
                    self.record_lock.acquire()
                    self.record_locked = True
                print("Sneaky Computer Time", flush = True)
                self.display_text("Training...")

                # Have the agent play a few rounds without showing to the user
                self.sneaky_train()

                self.clear_text(obs)
                self._increment_state()
            
            elif self.state is TRANSITION:
                if not self.record_locked:
                    self.record_lock.acquire()
                    self.record_locked = True
                self.transition("Computers Turn! Press <Space> to Start")
            
            elif self.state is COMPUTER_PLAY:
                if self.record_locked:
                    self.record_lock.release()
                    self.record_locked = False
                prev_obs, action, reward, obs, env_done = self._computer_play(obs)
            
            elif self.state is TRANSITION2:
                if not self.record_locked:
                    self.record_lock.acquire()
                    self.record_locked = True
                self.transition("Your Turn! Press <Space> to Start")

            # Increment the timer if it's the human or shown computer's turn
            if self.state is COMPUTER_PLAY or self.state is HUMAN_PLAY:
                if self.state == HUMAN_PLAY and isinstance(self.agent.memory, 'DQfDMemory'):
                    self.agent.memory.append_demonstration(prev_obs, action, reward, obs, env_done)
                else:
                    self.agent.memory.append(prev_obs, action, reward, obs, env_done)
                i += 1
                # Perform a quick learning process and increment the state after a certain time period has passed
                if i % (self.fps * self.seconds_play_per_state) == 0:
                    self.record_lock.acquire()
                    self.display_text("Demo Training...")
                    print("Begin Demonstration Training")
                    print("Number of transitions in buffer: ", len(self.agent.memory), flush = True)
                    for j in range(self.num_train_per_demo):
                        print("Iteration %d / %d" % (j + 1, self.num_train_per_demo))
                        self.agent.learn()
                    self.clear_text(obs)
                    self.record_lock.release()
                    self._increment_state()
                    i = 0
        
        # Stop the pygame environment when done
        pygame.quit()

