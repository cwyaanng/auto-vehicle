import os, sys
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')

import carla
import gym
import numpy as np
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
from env.wrapper import CarlaWrapperEnv

from stable_baselines3 import SAC 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ---------------- 설정 ----------------
DATA_DIR = "dataset_1"   
SIMULATION = "SAC_PURE_1M"
GLOBAL_SEED = 42

def make_env(batch_size):
    """
    CARLA 연결 후 CarlaWrapperEnv 반환
    Gym 0.21 API(reset->obs, step->(obs, reward, done, info)) 기준
    """
    client, world, carla_map = connect_to_carla()

    points = (0, 0)  # 필요하면 실제 웨이포인트로 교체

    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=points,
        simulation=SIMULATION,
        target_speed=22.0,
        tensorboard_log=SIMULATION+str(batch_size)
    )
    env = Monitor(env)
    try:
        env.seed(GLOBAL_SEED)
    except Exception:
        pass
    return env

def make_vec_env(batch_size):
    set_random_seed(GLOBAL_SEED)
    return DummyVecEnv([lambda: make_env(batch_size)])

def main(batch_size):
    # 1) CARLA 환경 준비
    env = make_vec_env(batch_size)

    # 3) 기본 SAC 모델 생성
    model = SAC(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=SIMULATION+str(batch_size),
        buffer_size=1_000_000,
        batch_size=batch_size,
        tau=0.005,
        verbose=1,
        seed=GLOBAL_SEED,
    )

    # 5) 온라인 학습 (커스텀 프리필/프리트레인 제거)
    model.learn(
        total_timesteps=1_000_000,
        log_interval=50,
        tb_log_name=f"{SIMULATION+str(batch_size)}"
    )

    # 6) 저장/마무리
    model.save(f"sac_{SIMULATION}_sb3_py37.zip")
    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            print("첫 번째 인자는 batch_size 정수여야 합니다.")
            sys.exit(1)
    else:
        batch_size = 256  # 기본값
    main(batch_size)
