from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer
from .q_network import DQN
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

def softmax(action_values, tau=1.0):
    """
    Uses softmax(x) = softmax(x-c) identity to resolve possible overflow from exponential of large numbers in softmax
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar. 
                     𝜏 is the temperature parameter which controls how much the agent focuses on the highest valued 
                     actions. The smaller the temperature, the more the agent selects the greedy action. 
                     Conversely, when the temperature is high, the agent selects among actions more uniformly random.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.array([max(preference)/tau for preference in action_values])
    
    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.array([np.exp(preference-max_preference) for preference, max_preference in zip(preferences, reshaped_max_preference)])
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = exp_preferences.sum(axis=1)
    
    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    
    # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()

    return action_probs

class Q_Agent(BaseAgent):
    def __init__(self):
        self.name = "Q-Learning Agent"
    
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.device = agent_config['device']
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                        agent_config['minibatch_size'],
                                        agent_config.get('seed'))
        self.network = DQN(agent_config['network_config']).to(self.device)     # The latest state of the network that is getting replay updates
        self.current_q = DQN(agent_config['network_config']).to(self.device)
        if agent_config.get('model_weights_load_path') is not None:
            self.network.load_state_dict(torch.load(agent_config.get('model_weights_load_path')))
            self.current_q.load_state_dict(torch.load(agent_config.get('model_weights_load_path')))
        # self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        optim_config = agent_config['optimizer_config']
        self.optimizer = optim.Adam(self.network.parameters(), lr=optim_config['step_size'], betas=optim_config['betas'])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get('seed'))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.epsiode_steps = 0

    def optimize_network(self, experiences):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
            discount (float): The discount factor.
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        """
        
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = torch.tensor(np.concatenate(states)).to(self.device)
        next_states = torch.tensor(np.concatenate(next_states)).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.tensor(terminals).to(self.device)
        batch_size = states.shape[0]
        batch_indices = np.arange(batch_size)
        state_action_values = self.network(states)[batch_indices, actions]

        next_state_action_values = self.current_q(next_states).max(1)[0].detach() * (1-terminals) # Q-learning
        expected_state_action_values = next_state_action_values * self.discount + rewards

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.float())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def policy(self, state):
        """
        Args:
            state (Numpy array): the state
        Returns:
            the action
        """
        state = torch.tensor(state).to(self.device)
        action_values = self.network(state).cpu().detach().numpy()
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        action = self.policy(state)
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()     
                self.optimize_network(experiences)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action
        
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences)
                

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")