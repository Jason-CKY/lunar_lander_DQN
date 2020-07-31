from Environments.environment import Environment
from Agents.q_agent import Q_Agent as Agent
from rl_glue import RLGlue
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gym

def get_average(agent_sum_reward, num=50):
    agent_sum_reward = np.array(agent_sum_reward)
    if len(agent_sum_reward) < num:
        return agent_sum_reward.mean()
    else:
        return agent_sum_reward[-num:].mean()

def savefig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name, env_name):
    np.save(npy_path, np.vstack([agent_sum_reward, average_sum_reward]))
    x = np.load(npy_path)
    plt.title("{} on {}".format(agent_name, env_name))
    plt.plot(np.arange(x.shape[1]), x[0], color='blue')
    plt.plot(np.arange(x.shape[1]), x[1], color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(['Actual Reward', 'Running Average'])
    plt.savefig(fig_path)

def update_agent_parameters(environment_parameters, agent_parameters):
    env = gym.make(environment_parameters['gym_environment'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, environment_parameters['gym_environment'], agent_parameters['name'])
    os.makedirs(agent_parameters['checkpoint_dir'], exist_ok=True)
    return agent_parameters


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = []
    average_sum_reward = []

    env_info = environment_parameters
    agent_info = agent_parameters

    rl_glue.rl_init(agent_info, env_info)

    starting_episode = 0

    gym_name = env_info['gym_environment']
    agent_name = agent_info['name']
    save_name = "{}.npy".format(rl_glue.agent.name)
    npy_path = os.path.join(rl_glue.agent.checkpoint_dir, "sum_reward_{}".format(save_name))
    fig_path = os.path.join(rl_glue.agent.checkpoint_dir, 'sum_rewards.png')

    # load checkpoint if any
    if experiment_parameters['load_checkpoint'] is not None:
        rl_glue.agent.load_checkpoint(experiment_parameters['load_checkpoint'])
        agent_sum_reward, average_sum_reward = np.load(npy_path)
        agent_sum_reward = list(agent_sum_reward)
        average_sum_reward = list(average_sum_reward)
        fname = experiment_parameters['load_checkpoint'].split(os.path.sep)[-1]
        try:
            starting_episode = int(fname.split('_')[1])
        except IndexError:
            starting_episode = len(agent_sum_reward)

        print(f"starting from episode {starting_episode}")

    
    for episode in tqdm(range(1 + starting_episode, experiment_parameters["num_episodes"]+1)):
        # run episode
        rl_glue.rl_episode(experiment_parameters["timeout"])

        episode_reward = rl_glue.rl_agent_message("get_sum_reward")
        agent_sum_reward.append(episode_reward)
        if episode % experiment_parameters['print_freq'] == 0:
            print('Episode {}/{} | Reward {}'.format(episode, experiment_parameters['num_episodes'], episode_reward))

        average = get_average(agent_sum_reward)
        average_sum_reward.append(average)

        if episode % experiment_parameters['checkpoint_freq'] == 0:
            rl_glue.agent.save_checkpoint(episode)
            savefig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name, gym_name)
        
        if env_info['solved_threshold'] is not None and average >= env_info['solved_threshold']:
            print("Task Solved with reward = {}".format(episode_reward))
            rl_glue.agent.save_checkpoint(episode, solved=True)
            break

    savefig(agent_sum_reward, average_sum_reward, npy_path, fig_path, agent_name, gym_name)

def main():
    # Run Experiment

    # Experiment parameters
    experiment_parameters = {
        "num_episodes" : 500,
        "checkpoint_freq": 100,
        "print_freq": 1,
        "load_checkpoint": None,
        # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
        # some number of timesteps.
        "timeout" : 1600
    }

    # Environment parameters
    environment_parameters = {
        "gym_environment": 'LunarLander-v2',
        'solved_threshold': 200,
        'seed': 0
    }

    current_env = Environment

    # Agent parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_parameters = {
        'network_config': {
            'state_dim': 8,
            'num_hidden_units': 256,
            'action_dim': 4,
            'seed': 0
        },
        'optimizer_config': {
            'step_size': 1e-3,
            'betas': (0.9, 0.999)
        },
        'name': 'q-learning',
        'device': device,
        'replay_buffer_size': 50000,
        'minibatch_size': 64,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001,
        'checkpoint_dir': 'model_weights',
        'seed': 0
    }
    current_agent = Agent
    agent_parameters = update_agent_parameters(environment_parameters, agent_parameters)
    # run experiment
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

if __name__ == '__main__':
    main()