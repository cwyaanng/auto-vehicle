import os, sys
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla
import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv

SIMULATION = "test_sac_pretrain"  
def make_env(batch_size=256):
    client, world, carla_map = connect_to_carla()
    points = (0, 0)  
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=points,
        simulation=SIMULATION,
        target_speed=22.0,
        tensorboard_log=SIMULATION+str(batch_size),
    )
    env = Monitor(env)
    try:
        env.seed(42)
    except Exception:
        pass
    return env

def make_vec_env(batch_size=256):
    return DummyVecEnv([lambda: make_env(batch_size)])

def evaluate(model_path,
             episodes=200,
             max_steps_per_episode=100_000,
             deterministic=True,
             batch_size=256,
             render=True):
    env = make_vec_env(batch_size)

    model = SAC.load(model_path, env=env, device="auto")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            if dones[0]:
                print(f"[Episode {ep}] steps={step+1}, return={ep_reward:.2f}")
                break
        else:
            print(f"[Episode {ep}] MAX steps reached, return={ep_reward:.2f}")

    env.close()

if __name__ == "__main__":
    MODEL_PATH = "/home/wise/chaewon/models/sac_SAC_7_1M_pretrained_finetuned_py37.zip"
    evaluate(MODEL_PATH, episodes=100, deterministic=True, batch_size=256)
