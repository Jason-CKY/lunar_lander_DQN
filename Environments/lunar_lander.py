from .base_environment import BaseEnvironment
import numpy as np
import gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        '''
        Setup for the environment called when the environment first starts
        '''
        self.env = gym.make("LunarLander-v2")
        freq = env_info.get("record_frequency")
        if freq is None:
            freq = 1
        self.env = gym.wrappers.Monitor(self.env, env_info['recording_dir'], video_callable=lambda episode_id: ((episode_id+1)%freq==0 or episode_id==0))
        self.env.seed(0)
    
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        reward = 0.0
        observation = self.env.reset()
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        # returns first state observation from the environment
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term
        
    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass
    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """
        pass