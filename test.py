import gym
from Agents.q_agent import Q_Agent
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4,
        'seed': 0
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8,
        'betas': (0.9, 0.999)
    },
    'name': 'q-learning agent',
    'device': device,
    'replay_buffer_size': 50000,
    'minibatch_size': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}



# env = gym.make('CartPole-v0')
# env = gym.make('Copy-v0')
# env = gym.make('BipedalWalker-v3')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Asteroids-v0')
env = gym.make("LunarLander-v2")
agent = Q_Agent()
last_state = env.reset()
last_state = np.array([last_state])
agent.agent_init(agent_parameters)
done = False
# agent.agent_start(state)
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action) # take a random action
    state = np.array([state])
    agent.replay_buffer.append(last_state, action, reward, 1, state)
    # agent.agent_step(reward, state)
    last_state = state

# agent.agent_end(reward)
env.close()
experiences = agent.replay_buffer.sample()
states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
states = torch.tensor(np.concatenate(states)).to('cuda')
next_states = torch.tensor(np.concatenate(next_states)).to('cuda')
terminals = torch.tensor(terminals).to(device)
rewards = torch.tensor(rewards).to(device)
# print(experiences)
# print(states)
# print(np.concatenate(states).shape)
next_state_action_values = agent.current_q(next_states).detach()
probabilities = agent.softmax(next_state_action_values.cpu().numpy(), agent.tau)
print(next_state_action_values.shape)
print(probabilities.shape)
out = next_state_action_values*torch.tensor(probabilities).to(device)
print(out)
print(out.sum(1))
print(out.sum(1).float())