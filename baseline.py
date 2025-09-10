import os
import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset, ReplayBuffer
from env.wrapper import CarlaWrapperEnv
from env.env_set import connect_to_carla
from datetime import datetime


# === 1. 오프라인 데이터 불러오기 (dataset_1/*.npz) ===
datasets = []
for file in os.listdir("dataset_1"):
    if file.endswith(".npz"):
        path = os.path.join("dataset_1", file)
        datasets.append(MDPDataset.load(path))

dataset = MDPDataset.concatenate(datasets)
print("Dataset size:", len(dataset))

# === 2. 알고리즘 정의 ===
cql = CQL(use_gpu=True)

# === 3. Offline Pretraining ===
print("🔹 Offline Pretraining 시작")
cql.fit(
    dataset,
    n_epochs=100,
    scorers={
        "td_error": d3rlpy.metrics.td_error_scorer,
        "value_mean": d3rlpy.metrics.average_value_estimation_scorer,
    },
)

# === 4. Online Fine-Tuning ===
print("🔹 Online Fine-Tuning 시작")

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
client, world, carla_map = connect_to_carla()
env = CarlaWrapperEnv(
    client=client,
    world=world,
    carla_map=carla_map,
    points=(0, 0),
    simulation="logs/CQL_FINE_TUNE/"+NOW
)

buffer = ReplayBuffer(capacity=5_000_000, env=env)

cql.fit_online(
    env,
    buffer,
    n_steps=200000,
    n_steps_per_epoch=1000,
    eval_interval=5000,
)

# === 5. 모델 저장 ===
cql.save_model("cql_offline_online.d3")
print("✅ Fine-Tuning 완료, 모델 저장됨.")
