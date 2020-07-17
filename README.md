# Pytorch implementation of deep Q-learning on the openAI lunar lander environment
Q-learning agent is tasked to learn the task of landing a spacecraft on the lunar surface.

Environment is provided by the openAI gym [1](https://gym.openai.com/envs/LunarLander-v2/)

Base environment and agent is written in RL-Glue standard [2](http://www.jmlr.org/papers/v10/tanner09a.html), providing the library and abstract classes to inherit from for reinforcement learning experiments.

## Updates:
* Added expected sarsa functionality. Change agent_parameter['name'] to either 'q-learning agent' or 'expected sarsa agent' for each type of learning algorithm.

## Results
<table align='center'>
<tr align='center'>
<td> type of agent </td>
<td> reward sum for each episode </td>
<td> last episode </td>
</tr>
<tr>
<td> Q learning agent </td>
<td><img src = 'images\q_learning_sum_rewards.png'> 
<td><img src = 'images\q_learning_episode_500.gif'>
</tr>
<tr>
<td> Expected sarsa agent </td>
<td><img src = 'images\expected_sarsa_sum_rewards.png'> 
<td><img src = 'images\expected_sarsa_episode_500.gif'>
</tr>
</table>

## Lunar Lander Environment
### Rewards
The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

### State
* s[0] is the horizontal coordinate
* s[1] is the vertical coordinate
* s[2] is the horizontal speed
* s[3] is the vertical speed
* s[4] is the angle
* s[5] is the angular speed
* s[6] 1 if first leg has contact, else 0
* s[7] 1 if second leg has contact, else 0

### Actions
Four discrete actions available: 
0: do nothing
1: fire left orientation engine 
2: fire main engine
3: fire right orientation engine

## Implementation Details
```
experiment_parameters = {
    "num_runs" : 1,
    "num_episodes" : 10,
    # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
    # some number of timesteps.
    "timeout" : 5000
}
environment_parameters = {
    "record_frequency": 10,
    "episode_dir": "episodes"
}
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'lr': 1e-3,
        'betas': (0.9, 0.999)
    },
    'name': 'expected sarsa agent',
    'device': 'cuda',
    'replay_buffer_size': 50000,
    'minibatch_size': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
```

## [Self implemented Softmax](https://github.com/Jason-CKY/lunar_lander_DQN/blob/7c0a3a8f581a11128acb9225791563a24a3db10f/Agents/q_agent.py#L199-L239)

<img src="images\softmax_equation.PNG"
     alt="Softmax_Equation" />

Implemented own softmax equation to avoid overflow problems from taking exponential of large numbers, using the softmax(x) = softmax(x-c) identity. 
𝜏 is the temperature parameter which controls how much the agent focuses on the highest valued actions. The smaller the temperature, the more the agent selects the greedy action. Conversely, when the temperature is high, the agent selects among actions more uniformly random.

## Dependencies
* [Instructions for installing openAI gym environment in Windows](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30)
* Tqdm
* ffmpeg (conda install -c conda-forge ffmpeg)
* pytorch (conda install pytorch torchvision cudatoolkit=10.2 -c pytorch)
* numpy

## How to use

### Training model for lunar lander environment
```
git clone https://github.com/Jason-CKY/lunar_lander_DQN.git
cd lunar_lander_DQN
Edit experiment parameters in main.py
python main.py
```