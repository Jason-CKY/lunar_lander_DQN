import gym
import torch
import numpy as np
import os
from Agents.q_agent import Q_Agent
import argparse
from PIL import Image

def update_agent_parameters(env_name, agent_parameters):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, env_name, agent_parameters['name'])
    os.makedirs(agent_parameters['checkpoint_dir'], exist_ok=True)

    return agent_parameters


def get_checkpoint_path(opt, checkpoint_dir):
    path = ''
    if opt.checkpoint is not None:
        path = os.path.join(checkpoint_dir, opt.checkpoint)
    else:
        if os.path.isfile(os.path.join(checkpoint_dir, 'solved.pth')):
            path = os.path.join(checkpoint_dir, 'solved.pth')
        else:
            files = [fname for fname in os.listdir(checkpoint_dir) if fname.endswith('.pth')]
            latest_episode = max([int(fname.split('_')[1]) for fname in files])
            for fname in files:
                if int(fname.split('_')[1]) == latest_episode:
                    path = os.path.join(checkpoint_dir, fname)
    
    if path == '':
        raise OSError('No checkpoint found')
    
    return path

def main(opt):
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
        'name': opt.agent,
        'device': device,
        'replay_buffer_size': 50000,
        'minibatch_size': 64,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001,
        'checkpoint_dir': 'model_weights',
        'seed': 0
    }

    env = gym.make(opt.env)

    agent_parameters = update_agent_parameters(opt.env, agent_parameters)
    agent = Q_Agent()
    agent.agent_init(agent_parameters)
    checkpoint_dir = agent_parameters['checkpoint_dir']
    
    if opt.gif:
        env = gym.wrappers.Monitor(env, checkpoint_dir, force=True)

    checkpoint_path = get_checkpoint_path(opt, checkpoint_dir)
    agent.load_checkpoint(checkpoint_path)
    last_state = env.reset()
    action = agent.policy(last_state)
    done = False
 
    while not done: 
        state, reward, done, info = env.step(action)
        action = agent.policy(state)
        last_state = state
        env.render()

    env.close()
    
    if opt.gif:
        from moviepy.editor import VideoFileClip
        mp4_file = [fname for fname in os.listdir(checkpoint_dir) if fname.endswith('.mp4')][0]
        mp4_file = os.path.join(checkpoint_dir, mp4_file)
        clip = VideoFileClip(mp4_file)
        clip.write_gif(os.path.join(checkpoint_dir, 'recording.gif'))
        junk_files = [fname for fname in os.listdir(checkpoint_dir) if fname.endswith('.mp4') or fname.endswith('.json')]
        for junk in junk_files:
            fname = os.path.join(checkpoint_dir, junk)
            os.remove(fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2", help="Environment name")
    parser.add_argument("--agent", type=str, default="q-learning", help="Agent name")
    parser.add_argument("--checkpoint", type=str, help="Name of checkpoint.pth file under model_weights/env/")
    parser.add_argument("--gif", action='store_true', help='Save rendered episode as a gif to model_weights/env/recording.gif')
    opt = parser.parse_args()

    main(opt)
