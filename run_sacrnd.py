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
from datetime import datetime 

DATA_DIR = "dataset_1"  
SIMULATION = "SAC_RND_TESTING_CRITIC"
NOW = ""
def make_env(batch_size):
    NOW = str(datetime.now())
    # carla 연결 
    client, world, carla_map = connect_to_carla()
    # 주행 시작 포인트 
    start_point = (0, 0)
    
    # 강화학습 환경 생성 
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=start_point,
        simulation=SIMULATION,
        logs = "logs/"+SIMULATION+"/"+NOW,
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
    # 시뮬레이션 & 강화학습 환경 생성 
    env = make_vec_env(batch_size)
    
    # 강화학습 모델 생성 
    trainer = SACOfflineOnline(env=env, buffer_size=1_000_000, batch_size=batch_size, tau=0.005, verbose=1, tensorboard_log="logs/"+SIMULATION+"/"+NOW)
    
    obs_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    rnd = RND(obs_dim, lr=1e-3, device=str(trainer.device))
   
   # pretrain with only route 6 data 
    trainer.prefill_from_npz_folder(DATA_DIR)
    trainer.pretrain_mcnet_supervised(steps=500000)
    trainer.attach_rnd(rnd)
    trainer.pretrain_critic(steps=1000000)  
    trainer.pretrain_actor(steps=1000000)
    
    trainer.save(f"pretrained_actor_critic_1M.zip")
    trainer.online_learn(log_interval=50, total_timesteps=1_000_000, tb_log_name=SIMULATION+str(batch_size))

    trainer.save(f"trained_1M_1M.zip")
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
