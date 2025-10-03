import os, sys

from agents.rnd import RND

sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla  
import gym
import numpy as np
from agents.sacrnd_model import SACOfflineOnline
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
from env.wrapper import CarlaWrapperEnv 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

DATA_DIR = "dataset_1"  
SIMULATION = "SAC_RND_TESTING_CRITIC"

def make_env(batch_size):
    client, world, carla_map = connect_to_carla()
    
    points = (0, 0)
    
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=points,
        simulation=SIMULATION,
        target_speed=22.0
    )
    env = Monitor(env)
    try:
        env.seed(42)
    except Exception:
        pass
    return env

def make_vec_env(batch_size):
    return DummyVecEnv([lambda: make_env(batch_size)])


def main(batch_size):
    env = make_vec_env(batch_size)
    
    trainer = SACOfflineOnline(env=env, buffer_size=1_000_000, batch_size=batch_size, tau=0.005, verbose=1, tensorboard_log=SIMULATION+str(batch_size))
    
    obs_dim = env.observation_space.shape[0]
    rnd = RND(obs_dim, lr=1e-3, device=str(trainer.device))
   
    
    trainer.prefill_from_npz_folder(DATA_DIR)
    trainer.attach_rnd(rnd)
    trainer.compute_mc_returns_from_buffer(gamma=trainer.gamma)
    trainer.pretrain_critic(steps=50000)  
    trainer.pretrain_actor(steps=50000)
    trainer.online_learn(log_interval=50, total_timesteps=1_000_000, tb_log_name=SIMULATION+str(batch_size))

    trainer.save(f"sac_{SIMULATION+str(batch_size)}_pretrained_finetuned_py37.zip")
    env.close()

if __name__ == "__main__":    
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            print("첫 번째 인자는 batch_size 정수여야 합니다.")
            sys.exit(1)
    else:
        batch_size = 256 

    main(batch_size)
