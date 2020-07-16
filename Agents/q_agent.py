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
        self.network = DQN(agent_config['network_config']).to(self.device)      # The latest state of the network that is getting replay updates
        self.current_q = DQN(agent_config['network_config']).to(self.device)    # The fixed network used for computing the targets, and particularly, the action-values at the next-states.
        optim_config = agent_config['optimizer_config']
        # self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.optimizer = optim.Adam(self.network.parameters(), lr=optim_config['lr'], betas=optim_config['betas'])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get('seed'))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.epsiode_steps = 0

    def policy(self, state):
        """
        Args:
            state (Numpy array): the state
        Returns:
            the action
        """
        action_values = self.network(torch.tensor(state).to(self.device)).cpu().detach().numpy()
        probabilities = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probabilities)
        return action

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.epsiode_steps = 0
        self.last_state = observation
        self.last_action = self.policy(np.array([self.last_state]))
        return self.last_action

    def agent_step(self, reward, observation):
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
        self.epsiode_steps += 1

        # state = np.array([observation])
        state = observation
        action = self.policy(np.array([state]))

        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)

        # perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                states, actions, rewards, terminals, next_states = zip(*experiences)
                # states = np.array(states)
                states = torch.tensor(states).to(self.device)
                # print(states)
                next_states = torch.tensor(next_states).to(self.device)
                state_action_values = self.network(states)
                # print(state_action_values.dtype())
                next_state_action_values = self.current_q(next_states).cpu().detach().numpy()
                next_state_action_values = next_state_action_values.max(axis=1) * (1-np.array(terminals))
                bootstrap_return = rewards + self.discount*next_state_action_values

                # batch_indices = np.arange(state_action_values.shape[0])
                # state_action_values = state_action_values[batch_indices, actions]
                state_action_values = state_action_values.gather(1, torch.tensor(actions).to(self.device).unsqueeze(1))

                # Compute Huber Loss
                # state_action_values = torch.tensor(state_action_values).unsqueeze(1).to(device)
                bootstrap_return = torch.tensor(bootstrap_return).unsqueeze(1).to(self.device).float()
                loss = F.smooth_l1_loss(state_action_values, bootstrap_return)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.network.parameters():
                    param.grad.data.clamp_(-1, 1)   # gradient clipping to prevent explosion of gradients (Deepmind 205 DQN)
                self.optimizer.step()

        self.last_state = observation
        self.last_action = action

        return action
        
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.sum_rewards += reward
        self.epsiode_steps += 1
        
        state = np.zeros_like(self.last_state)
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)

        # perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                states, actions, rewards, terminals, next_states = zip(*experiences)
                states = torch.tensor(states).to(self.device)
                next_states = torch.tensor(next_states).to(self.device)
                state_action_values = self.network(states)
                next_state_action_values = self.current_q(next_states).cpu().detach().numpy()
                next_state_action_values = next_state_action_values.max(axis=1) * (1-np.array(terminals))
                bootstrap_return = rewards + self.discount*next_state_action_values

                # batch_indices = np.arange(state_action_values.shape[0])
                # state_action_values = state_action_values[batch_indices, actions]
                state_action_values = state_action_values.gather(1, torch.tensor(actions).to(self.device).unsqueeze(1))

                # Compute Huber Loss
                # state_action_values = torch.tensor(state_action_values).unsqueeze(1).to(device)
                bootstrap_return = torch.tensor(bootstrap_return).unsqueeze(1).to(self.device).float()
                loss = F.smooth_l1_loss(state_action_values, bootstrap_return)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.network.parameters():
                    param.grad.data.clamp_(-1, 1)   # gradient clipping to prevent explosion of gradients (Deepmind 205 DQN)
                self.optimizer.step()

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