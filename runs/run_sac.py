import os, sys

sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla  
import gym
import numpy as np
from agents.sac_model import SACOfflineOnline
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
from env.wrapper import CarlaWrapperEnv 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

DATA_DIR = "dataset_1"  

SIMULATION = "SAC_10_1M"

def make_env():

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
  
    trainer = SACOfflineOnline(
        env=env,
        tensorboard_log=env.log,
        buffer_size=1_000_000,
        batch_size=batch_size,
        tau=0.005,
        verbose=1,
    )

    trainer.prefill_from_npz_folder(DATA_DIR)
    trainer.pretrain_critic(steps=50000)
    trainer.pretrain_actor(steps=50000)
    print("online fine tuning start")
    trainer.online_learn(total_timesteps=1_000_000, tb_log_name=f"{SIMULATION+str(batch_size)}")
    trainer.save(env.log+"/model/"+f"{SIMULATION}.zip")
    env.close()

if __name__ == "__main__":    
    if len(sys.argv) > 1:
        try:
            SIMULATION = str(sys.argv[1])
            batch_size = int(sys.argv[2])
        except ValueError:
            print("입력 형식은 python {파일명} {시뮬레이션 이름} {배치 사이즈}")
            sys.exit(1)
    else:
        batch_size = 256  

    main(batch_size)
