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
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        simulation="logs/"+SIMULATION+"/"+NOW,
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
    trainer = SACOfflineOnline(env=env, buffer_size=4_000_000, batch_size=batch_size, tau=0.005, verbose=1, tensorboard_log="logs/"+SIMULATION+"/"+NOW)
    
    obs_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    rnd = RND(obs_dim, lr=1e-3, device=str(trainer.device))
   
    print("모든 데이터 버퍼에 저장")
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    print("critic pretrain => warm start")
    trainer.attach_rnd(rnd)
    trainer.pretrain_critic()
    print("모든 주행 데이터 actor behavioral cloning")
    trainer.pretrain_actor(100000)
    
    
    print("여러 주행 데이터로 mcnet 학습")  
    trainer.replay_buffer.reset()
    print("data filling start")
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    print("mcnet 학습중")
    trainer.train_mcnet_from_buffer(epochs=30)
    print("mcnet 모델 저장")
    trainer.save_mcnet_pth(f"mcnet/mcnet_pretrained.pth")
    trainer.save_mcnet_pickle(f"mcnet/mcnet_pretrained.pkl")
    
    print("모든 주행 데이터 넣음")
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    trainer.save(f"pretrained_actor_critic_1M.zip")
    trainer.online_learn(log_interval=50, total_timesteps=1_000_000, tb_log_name="logs/"+SIMULATION+"/"+NOW)

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
