import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rlkit.envs.images import EnvRenderer, InsertImageEnv
import rlkit.torch.pytorch_util as ptu
#from rlkit.envs.encoder_wrappers import VQVAEWrappedEnv
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--downsample", action='store_true')
parser.add_argument("--test_env_seeds", nargs='+', type=int)
parser.add_argument("--full_open_close_init_and_goal", action="store_true")
parser.add_argument("--fix_drawer_orientation", action="store_true")
parser.add_argument("--fix_drawer_orientation_semicircle", action='store_true')
parser.add_argument("--drawer_sliding", action='store_true')
parser.add_argument("--red_drawer_base", action='store_true')
parser.add_argument("--new_view", action='store_true')
parser.add_argument("--close_view", action='store_true')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()

test_env_seeds = args.test_env_seeds[5:6] if args.debug else args.test_env_seeds
num_trajectories = 2 if args.debug else args.num_trajectories
for test_env_seed in test_env_seeds:
    if args.debug:
        data_save_path = "/2tb/home/patrickhaoy/data/test/test.pkl"
    else:
        data_save_path = "/2tb/home/patrickhaoy/data/affordances/data/new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer/{}top_drawer_goals_seed{}.pkl".format("full_open_close_" if args.full_open_close_init_and_goal else "", str(test_env_seed))

    kwargs = {
        'test_env_seed': test_env_seed,
        'full_open_close_init_and_goal': args.full_open_close_init_and_goal,
        'expl': True if args.full_open_close_init_and_goal else False,
        'fix_drawer_orientation': True if args.fix_drawer_orientation else False,
        'fix_drawer_orientation_semicircle': True if args.fix_drawer_orientation_semicircle else False,
        'drawer_sliding': True if args.drawer_sliding else False,
        'new_view': True if args.new_view else False,
        'close_view': True if args.close_view else False,
        'red_drawer_base': True if args.red_drawer_base else False,
        'gui': True if args.gui else False
    }
    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196
    env = roboverse.make('SawyerRigAffordances-v1', test_env=True, **kwargs)


    obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size
    imlength = env.obs_img_dim * env.obs_img_dim * 3

    dataset = {
            'initial_latent_state': np.zeros((num_trajectories * args.num_timesteps, 720), dtype=np.float),
            'latent_desired_goal': np.zeros((num_trajectories * args.num_timesteps, 720), dtype=np.float),
            'state_desired_goal': np.zeros((num_trajectories * args.num_timesteps,
                obs_dim), dtype=np.float),
            'image_desired_goal': np.zeros((num_trajectories * args.num_timesteps, imlength), dtype=np.float),
            'initial_image_observation': np.zeros((num_trajectories * args.num_timesteps, imlength), dtype=np.float),
            }

    for i in tqdm(range(num_trajectories)):
        env.demo_reset()
        init_img = np.uint8(env.render_obs()).transpose() / 255.0
        
        for k in range(40):
            action = env.get_demo_action(first_timestep=(k == 0), final_timestep=False)
            obs, reward, done, info = env.step(action)
        
        for j in range(args.num_timesteps):
            action = env.get_demo_action(first_timestep=False, final_timestep=(j == args.num_timesteps-1))
            obs, reward, done, info = env.step(action)

            img = np.uint8(env.render_obs()).transpose() / 255.0

            dataset['state_desired_goal'][i * args.num_timesteps + j] = obs['state_achieved_goal']
            dataset['image_desired_goal'][i * args.num_timesteps + j] = img.flatten()
            dataset['initial_image_observation'][i * args.num_timesteps + j] = init_img.flatten()

    file = open(data_save_path, 'wb')
    pkl.dump(dataset, file)
    file.close()