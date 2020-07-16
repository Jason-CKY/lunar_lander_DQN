from Environments.lunar_lander import LunarLanderEnvironment
from Agents.agent import Q_Agent as Agent
from rl_glue import RLGlue
import numpy as np
import torch
from tqdm import tqdm
import os

def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    rl_glue = RLGlue(environment, agent)
        
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], 
                                 experiment_parameters["num_episodes"]))

    env_info = environment_parameters

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)
        
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    path = os.path.join("results", "sum_reward_{}".format(save_name))
    np.save(path, agent_sum_reward)

# Run Experiment

# Experiment parameters
experiment_parameters = {
    "num_runs" : 1,
    "num_episodes" : 10,
    # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
    # some number of timesteps. Here we use the default of 1000.
    "timeout" : 1000
}

# Environment parameters
environment_parameters = {
    "record_frequency": 10
}

current_env = LunarLanderEnvironment

# Agent parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'hidden_dim': 128,
        'num_actions': 4
    },
    'optimizer_config': {
        'lr': 1e-3,
        'betas': (0.9, 0.999)
    },
    'device': device,
    'replay_buffer_size': 50000,
    'minibatch_size': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
current_agent = Agent

# run experiment
run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)