import roboverse
from PIL import Image
import numpy as np
import os
import pickle

sawyer_state_env = roboverse.make('SawyerRigAffordances-v6')

def look_at_demo(i=0, traj=0, steps=[0, 20, 40, 60, 70]):
    demo_path = f'/media/4tb/jason/pybullet_data/test_demos_{i}.pkl'
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)
    demo_obs = demo[traj]['observations']

    imshape = (3, 48, 48)
    transpose = (2, 1, 0)

    for step in steps:
        demo_img = demo_obs[step]['image_observation'].reshape(imshape)
        demo_img = (np.transpose(demo_img, transpose) * 255).astype(np.uint8)
        demo_im = Image.fromarray(demo_img)

        save_folder = '/home/jason/pybullet/fiddling_imgs/'
        demo_img_name = f'demo_traj{traj}-step{step}.png'
        demo_im.save(os.path.join(save_folder, demo_img_name))
    
    return demo

# there are these image npy files that are also saved, but i noticed they are identical to the 'image_observations' in the demo observations
def look_at_images(i=0, traj=0, steps=[0, 20, 40, 60, 70]):
    image_path = f'/media/4tb/jason/pybullet_data/test_images_{i}.npy'
    
    image_dict = np.load(image_path, allow_pickle=True).item()
    obs_imgs = image_dict['observations']
    
    imshape = (3, 48, 48)
    transpose = (2, 1, 0)

    for step in steps:
        img = obs_imgs[traj][step].reshape(imshape)
        img = np.transpose(img, transpose)
        im = Image.fromarray(img)

        save_folder = '/home/jason/pybullet/fiddling_imgs/'
        img_name = f'traj{traj}-step{step}.png'
        im.save(os.path.join(save_folder, img_name))

demo = look_at_demo(i=0)