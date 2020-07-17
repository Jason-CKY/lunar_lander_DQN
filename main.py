from Environments.lunar_lander import LunarLanderEnvironment
from Agents.q_agent import Q_Agent as Agent
from rl_glue import RLGlue
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    if not os.path.exists(environment_parameters['result_dir']):
        os.makedirs(environment_parameters['result_dir'])
    os.makedirs(os.path.sep.join(experiment_parameters['model_weights_save_path'].split(os.path.sep)[:-1]), exist_ok=True)
    
    path = os.path.join(environment_parameters['result_dir'], "sum_reward_{}".format(save_name))
    torch.save(rl_glue.agent.network.state_dict(), experiment_parameters['model_weights_save_path'])
    np.save(path, agent_sum_reward)
    x = np.load(path)
    plt.plot(np.arange(experiment_parameters['num_episodes']), x[0])
    plt.savefig(f'{environment_parameters['result_dir']}{os.path.sep}sum_rewards.png')

def main():
    # Run Experiment

    # Experiment parameters
    experiment_parameters = {
        "num_runs" : 1,
        "num_episodes" : 500,
        # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
        # some number of timesteps.
        "timeout" : 5000,
        'model_weights_save_path': f"results{os.path.sep}experiment.pt"
    }

    # Environment parameters
    environment_parameters = {
        "record_frequency": 500,
        "result_dir": "results",
        'recording_dir': f"results{os.path.sep}recording"
    }

    current_env = LunarLanderEnvironment

    # Agent parameters
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
            'betas': (0.9, 0.999)
        },
        'name': 'expected sarsa agent',
        'device': device,
        'replay_buffer_size': 50000,
        'minibatch_size': 8,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001
    }
    current_agent = Agent
    # os.makedirs(os.path.sep.join(experiment_parameters['model_weights_save_path'].split(os.path.sep)[:-1]), exist_ok=True)
    # run experiment
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

if __name__ == '__main__':
    main()