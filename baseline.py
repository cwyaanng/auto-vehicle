# baseline.py  (d3rlpy 0.x 호환 버전)

import sys
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')  # CARLA egg
import os
import numpy as np
import carla  # noqa: F401  # ensure the egg import is resolved

import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy import metrics

from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv
from datetime import datetime


# =======================
# Configs
# =======================
DATA_DIR = "dataset_1"
LOG_ROOT = "logs/CQL_FINE_TUNE_1"

DO_OFFLINE_PRETRAIN = False
PRETRAIN_STEPS = 50_000          
PRETRAIN_STEPS_PER_EPOCH = 5_000 

ONLINE_STEPS = 2_000_000
ONLINE_STEPS_PER_EPOCH = 10_000   

REPLAY_MAXLEN = 4_000_000
USE_GPU = True 


def build_mdpdataset_from_npz_dir(data_dir: str) -> MDPDataset:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    obs_chunks, act_chunks, rew_chunks, term_chunks = [], [], [], []
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files in {data_dir}")

    for fname in npz_files:
        path = os.path.join(data_dir, fname)
        d = np.load(path, allow_pickle=False)

        obs   = d["observations"].astype(np.float32)                # (N, obs_dim)
        acts  = d["actions"].astype(np.float32)                     # (N, act_dim)
        rews  = d["rewards"].astype(np.float32).reshape(-1)         # (N,1) -> (N,)
        terms = d["terminals"].astype(bool).reshape(-1)             # (N,1) -> (N,)

        # 키별 길이 다를 수 있으니 맞춰줌
        N = min(obs.shape[0], acts.shape[0], rews.shape[0], terms.shape[0])
        obs, acts, rews, terms = obs[:N], acts[:N], rews[:N], terms[:N]

        # 파일 경계에서 에피소드 단절(전 파일 마지막 스텝을 done=True로 강제)
        if term_chunks:
            term_chunks[-1][-1] = True

        obs_chunks.append(obs)
        act_chunks.append(acts)
        rew_chunks.append(rews)
        term_chunks.append(terms)

    observations = np.concatenate(obs_chunks, axis=0)
    actions      = np.concatenate(act_chunks, axis=0)
    rewards      = np.concatenate(rew_chunks, axis=0)
    terminals    = np.concatenate(term_chunks, axis=0)

    print("[Dataset] shapes:",
          "obs", observations.shape,
          "act", actions.shape,
          "rew", rewards.shape,
          "term", terminals.shape)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,  # d3rlpy 0.x: timeouts 인자 없음
    )
    print("[Dataset] size (episodes):", len(dataset))
    return dataset


def make_env(log_root: str):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    client, world, carla_map = connect_to_carla()
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=(0, 0),
        simulation=os.path.join(log_root, now),
    )
    return env


def main():
    # === 1) .npz → MDPDataset (d3rlpy 0.x 호환) ===
    dataset = build_mdpdataset_from_npz_dir(DATA_DIR)

    # === 2) 알고리즘 정의 ===
    cql = CQL(use_gpu=USE_GPU)

    # === 3) Offline Pretraining (스텝 기준) ===
    if DO_OFFLINE_PRETRAIN:
        print("Offline Pretraining 시작")
        try:
            cql.fit(
                dataset,
                n_steps=PRETRAIN_STEPS,
                n_steps_per_epoch=PRETRAIN_STEPS_PER_EPOCH,
                scorers={
                    "td_error": metrics.td_error_scorer,
                    "value_mean": metrics.average_value_estimation_scorer,
                },
            )
        except TypeError:
            epochs = max(1, PRETRAIN_STEPS // PRETRAIN_STEPS_PER_EPOCH)
            print(f"[Fallback] n_steps 파라미터 미지원 → n_epochs={epochs}로 대체")
            cql.fit(
                dataset,
                n_epochs=epochs,
                scorers={
                    "td_error": metrics.td_error_scorer,
                    "value_mean": metrics.average_value_estimation_scorer,
                },
            )

    # === 4) Online Fine-Tuning ===
    print("Online Fine-Tuning 시작")
    env = make_env(LOG_ROOT)
    buffer = ReplayBuffer(maxlen=REPLAY_MAXLEN, env=env)

    try:
        # 0.x에서 지원되는 표준 인자만 사용 (eval_interval은 0.x 일부 버전엔 없음)
        cql.fit_online(
            env,
            buffer,
            n_steps=ONLINE_STEPS,
            n_steps_per_epoch=ONLINE_STEPS_PER_EPOCH,
        )
    except TypeError:
        print("[Fallback] fit_online 인자 제한 → 최소 인자만 사용")
        cql.fit_online(
            env,
            buffer,
            n_steps=ONLINE_STEPS,
        )

    # === 5) 모델 저장 ===
    out_path = "cql_offline_online.d3"
    cql.save_model(out_path)
    print(f"Fine-Tuning 완료, 모델 저장 → {out_path}")


if __name__ == "__main__":
    # Gym 경고(NumPy 2.x) 안내만 출력
    try:
        import gym  # noqa: F401
        if int(np.__version__.split('.')[0]) >= 2:
            print("[Note] NumPy 2.x + gym 조합에서 경고가 발생할 수 있습니다. "
                  "가능하면 Gymnasium으로의 마이그레이션을 검토하세요.")
    except Exception:
        pass

    main()
