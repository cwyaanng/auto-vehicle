import os
import glob
import random
import csv
from pathlib import Path

import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATA_DIR   = "dataset_1"
STATE_KEY  = "observations"
ACTION_KEY = "actions"

EPOCHS     = 10
BATCH_SIZE = 2048
LR         = 1e-3
HID        = 256
OUT_DIM    = 128
DEVICE     = "cuda" if th.cuda.is_available() else "cpu"
SEED       = 42

CSV_OUT    = "rnd_novelty.csv"
TRAIN_RATIO = 0.8

from agents.rnd import RND


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

def peek_state_action_dim(file_path, state_key, action_key):
    d = np.load(file_path)
    S = d[state_key].shape[1]
    A = d[action_key].shape[1]
    return S, A

def load_npz_concat_state_action(file_path, state_key="state", action_key="action"):
    data = np.load(file_path)
    state = data[state_key]
    action = data[action_key]
    assert state.shape[0] == action.shape[0], f"Length mismatch in {file_path}"
    x = np.concatenate([state, action], axis=-1)
    return x.astype(np.float32)  

class NumpyArrayDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, idx):
        return self.arr[idx]

def train_rnd(model, train_loader, epochs=5, device="cpu"):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            loss = model.update(batch)
            total_loss += float(loss) * batch.size(0)
            n += batch.size(0)
        print(f"[Epoch {epoch}] train MSE to target = {total_loss / max(n,1):.6f}")


@th.no_grad()
def compute_novelty(model, x: np.ndarray, device="cpu", batch_size=8192):
    model.eval()
    scores = []
    N = x.shape[0]
    for i in range(0, N, batch_size):
        xb = th.from_numpy(x[i:i+batch_size]).float().to(device)
        nov = model.novelty(xb)  
        scores.append(nov.squeeze(1).cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([])

def clean_scores(arr: np.ndarray):
    return arr[np.isfinite(arr)]  # nan, inf 제거

def main():
    set_seeds(SEED)
    rng = np.random.default_rng(SEED)
    data_dir = Path(DATA_DIR)

    # 1) 파일 수집
    route6_files = sorted(glob.glob(str(data_dir / "route_6*.npz")))
    circle_route_files = sorted(glob.glob(str(data_dir / f"route_7*.npz")))
    assert len(route6_files) > 0, "no file for straight route."

    # 2) route_6 80/20 split
    idxs = list(range(len(route6_files)))
    random.shuffle(idxs)
    split = int(len(idxs) * TRAIN_RATIO)
    train_files = [route6_files[i] for i in idxs[:split]]
    valid_straight_files = [route6_files[i] for i in idxs[split:]]

    print(f"route_6 파일 수: {len(route6_files)} | 학습(80%): {len(train_files)} | 홀드아웃(20%): {len(valid_straight_files)}")
    print(f"테스트 파일 수: {len(valid_straight_files) + len(circle_route_files)}")

    # 3) 학습 데이터
    train_arrays = [load_npz_concat_state_action(fp, STATE_KEY, ACTION_KEY) for fp in train_files]
    X_train_raw = np.concatenate(train_arrays, axis=0)
    sample_obs_dim, sample_action_dim = peek_state_action_dim(train_files[0], STATE_KEY, ACTION_KEY)
    
    mu_state = X_train_raw[:,:sample_obs_dim].mean(axis=0)
    std_state = X_train_raw[:,:sample_obs_dim].std(axis=0) + 1e-8
    mu_action = X_train_raw[:,sample_obs_dim:].mean(axis=0)
    std_action = X_train_raw[:,sample_obs_dim:].std(axis=0) + 1e-8

    def apply_scaler(X):
        Xs = (X[:,:sample_obs_dim] - mu_state) / std_state
        Xa = (X[:, sample_obs_dim:] - mu_action )/ std_action
        return np.concatenate([Xs,Xa] , axis = -1).astype(np.float32)
    
    # 학습 데이터
    X_train = apply_scaler(X_train_raw)

    # 4) 모델 생성
    obs_dim = X_train.shape[1]
    model = RND(obs_dim=obs_dim, hid=HID, out_dim=OUT_DIM, lr=LR, device=DEVICE)
    print(model)

    # 5) 학습
    train_loader = DataLoader(NumpyArrayDataset(X_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_rnd(model, train_loader, epochs=EPOCHS, device=DEVICE)

    # 6) validation 데이터 novelty
    holdout_arrays = [load_npz_concat_state_action(fp, STATE_KEY, ACTION_KEY) for fp in valid_straight_files]
    X_holdout_raw = np.concatenate(holdout_arrays, axis=0) if holdout_arrays else np.zeros((0, obs_dim), dtype=np.float32)
    X_holdout = apply_scaler(X_holdout_raw)
    nov_holdout = clean_scores(compute_novelty(model, X_holdout, device=DEVICE))

    # 7) 노이즈 실험 (train 데이터에 noise 추가)
    noise_stds = [0.25, 0.5]
    nov_noise = {}
    for std in noise_stds:
        Xn = (X_train + rng.normal(0.0, std, size=X_train.shape)).astype(np.float32)
        nov_noise[std] = clean_scores(compute_novelty(model, Xn, device=DEVICE))

    # 8) circle route novelty
    circle_route_arrays = [load_npz_concat_state_action(fp, STATE_KEY, ACTION_KEY) for fp in circle_route_files]
    X_circle_route_raw = np.concatenate(circle_route_arrays, axis=0) if circle_route_arrays else np.zeros((0, obs_dim), dtype=np.float32)
    X_circle_route = apply_scaler(X_circle_route_raw)
    nov_circle_route = clean_scores(compute_novelty(model, X_circle_route, device=DEVICE))
    print(nov_circle_route)
    print(len(nov_circle_route))
    
    # --- 시각화 1: Validation vs Noise (route_6) ---
    plt.figure()
    plt.hist(nov_holdout, bins=100, density=True, alpha=0.6, label="Validation (route_6 holdout)")
    for std, nov in nov_noise.items():
        plt.hist(nov, bins=100, density=True, alpha=0.5, label=f"Train+Noise std={std}")
    plt.xlabel("Novelty")
    plt.ylabel("Density")
    plt.title("Novelty distributions: validation vs noise (route_6)")
    plt.legend()
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("novelty_hist_route6_noise.png", dpi=150)
    plt.close()
    print("[Saved] novelty_hist_route6_noise.png")

    # --- 시각화 2: Validation (route_6) vs Circle (route_7) ---
    plt.figure()

    # 공통 bin을 고정된 범위로 생성 (10^-2 ~ 10^5)
    bins = np.logspace(-5, 5, 200)

    plt.hist(nov_holdout, bins=bins, density=True, alpha=0.6, label="Validation (route_6 holdout)")
    plt.hist(nov_circle_route, bins=bins, density=True, alpha=0.6, label="Circle route (route_7)")

    plt.xscale("log")
    plt.xlim(1e-8, 1e5)  # x축 범위 고정
    plt.xlabel("Novelty")
    plt.ylabel("Density")
    plt.title("Novelty distributions: route_6 validation vs route_7")
    plt.legend()
    plt.tight_layout()
    plt.savefig("novelty_hist_route6_vs_route7_fixed.png", dpi=150)
    plt.close()
    print("[Saved] novelty_hist_route6_vs_route7_fixed.png")


    plt.figure()

    bins = np.logspace(-5, 5, 200)

    plt.hist(nov_holdout, bins=bins, alpha=0.6, label="Validation (route_6 holdout)")
    plt.hist(nov_circle_route, bins=bins, alpha=0.6, label="Circle route (route_7)")

    plt.xscale("log")
    plt.yscale("log")   # y축도 로그!
    plt.xlim(1e-8, 1e5)
    plt.xlabel("Novelty")
    plt.ylabel("Count (log scale)")
    plt.title("Novelty distributions: route_6 vs route_7")
    plt.legend()
    plt.tight_layout()
    plt.savefig("novelty_hist_route6_vs_route7_logy.png", dpi=150)
    plt.close()
    print("[Saved] novelty_hist_route6_vs_route7_logy.png")


if __name__ == "__main__":
    main()
