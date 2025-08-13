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

# ---------------- 데이터 로드 (경로만 넘기고, 실제 로드는 sac_model에서) ----------------
DATA_DIR = "dataset_1"   # *.npz 묶음 폴더
SIMULATION = "SAC_3"

def make_env():
    """
    CARLA 연결, 센서 부착, 차량 스폰 등 환경 준비 후
    gym.Env 형태로 반환하는 래퍼가 있으면 그걸 여기서 생성하세요.
    SB3 1.5.1은 gym==0.21 API 기준입니다 (reset()->obs, step()->obs, reward, done, info).
    """
    # 예시(가짜): 실제로는 당신의 CARLA Gym wrapper를 리턴
    client, world, carla_map = connect_to_carla()
    
    points = [
        (-150, -130),
        (100, 130),
        (60, 60)
    ]
    
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=points,
        simulation=SIMULATION,
        target_speed=22.0,
    )

    try:
        env.seed(42)
    except Exception:
        pass
    return env

def make_vec_env():
    # 단일 환경이면 DummyVecEnv로 감싸주면 SB3가 학습 가능
    return DummyVecEnv([make_env])


def main():
    # 1) CARLA 환경 준비
    env = make_vec_env()

    # 2) SAC offline→online 클래스 구성
    trainer = SACOfflineOnline(
        env=env,
        # 로깅은 wandb와 폴더 구조 잘 정립 필요 # 
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        verbose=1,
    )

    # 3) 오프라인 데이터로 버퍼 프리필
    trainer.prefill_from_npz_folder(DATA_DIR)

    # 4) 오프라인 critic → actor 순서로 사전학습
    trainer.pretrain_critic(steps=50000)
    trainer.pretrain_actor(steps=50000)
    print("파인튜닝 시작")
  
    trainer.online_learn(total_timesteps=300_000, tb_log_name="phase_online")

    # 6) 저장
    trainer.save("sac_offline_pretrained_finetuned_py37.zip")

    env.close()

if __name__ == "__main__":
    main()
