import os
import glob
import random
import csv
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
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
PLOT_OUT   = "rnd_plot.png"
TRAIN_RATIO = 0.8

from agents.rnd import RND


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

class FiLMEncoder(nn.Module):
    """
    obs_dim -> hid -> obs_dim (f_obs)
    act_dim -> hid -> (gamma, beta) each obs_dim
    z = gamma * f_obs(obs) + beta
    """
    def __init__(self, obs_dim: int, act_dim: int, hid: int = 256, device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim 
        self.f_obs = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, self.out_dim)
        ).to(device)
        self.f_act = nn.Sequential(
            nn.Linear(act_dim, hid), nn.ReLU(),
            nn.Linear(hid, 2 * self.out_dim) 
        ).to(device)
     
        for p in self.parameters():
            p.requires_grad = False
        self.device = device

    @th.no_grad()
    def transform(self, x_concat: np.ndarray, s_dim: int) -> np.ndarray:
        assert x_concat.ndim == 2 and x_concat.shape[1] == (s_dim + self.act_dim)
        obs = th.from_numpy(x_concat[:, :s_dim]).float().to(self.device)     
        act = th.from_numpy(x_concat[:, s_dim:]).float().to(self.device) 
        # 관측값을 latent feature h로 변환   
        h = self.f_obs(obs)                  
        # 행동에서 람다 / 베타 생성                               
        ga_be = self.f_act(act)                                             
        gamma, beta = ga_be.chunk(2, dim=-1)                                 
        z = gamma * h + beta                                                 
        return z.cpu().numpy().astype(np.float32)
      
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

    # train/valid/test => same scale 
    def apply_scaler(X):
        Xs = (X[:,:sample_obs_dim] - mu_state) / std_state
        Xa = (X[:, sample_obs_dim:] - mu_action )/ std_action
        return np.concatenate([Xs,Xa] , axis = -1).astype(np.float32)
    
    # 학습 데이터 생성 
    X_train = apply_scaler(X_train_raw)
    film = FiLMEncoder(obs_dim=sample_obs_dim, act_dim=sample_action_dim, hid=HID, device=DEVICE)
    Z_train = film.transform(X_train, s_dim=sample_obs_dim)  
    
    # 4) 모델 생성
    obs_dim = Z_train.shape[1]
    model = RND(obs_dim=obs_dim, hid=HID, out_dim=OUT_DIM, lr=LR, device=DEVICE)
    print(model)

    # 5) 학습
    train_loader = DataLoader(NumpyArrayDataset(Z_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_rnd(model, train_loader, epochs=EPOCHS, device=DEVICE)

    # compute novelty for validation data (straight data 20%)
    holdout_arrays, holdout_index_map = [], []
    for fp in valid_straight_files:
        x = load_npz_concat_state_action(fp, STATE_KEY, ACTION_KEY)
        holdout_arrays.append(x)
        holdout_index_map.extend([(os.path.basename(fp), i) for i in range(x.shape[0])])
        
    X_holdout_raw = np.concatenate(holdout_arrays, axis=0) if holdout_arrays else np.zeros((0, obs_dim), dtype=np.float32)
    X_holdout = apply_scaler(X_holdout_raw)
    Z_holdout = film.transform(X_holdout, s_dim=sample_obs_dim) if X_holdout.size else np.zeros((0, obs_dim), np.float32)
    nov_holdout = compute_novelty(model, Z_holdout, device=DEVICE) if Z_holdout.size else np.array([])
    
    # 학습 데이터에 노이즈 추가한 데이터 생성 
    noise_stds = [0.25, 0.5]
    X_train_noise = {
        std: (X_train + rng.normal(0.0, std, size=X_train.shape)).astype(np.float32)
        for std in noise_stds
    }
    
    # 노이즈를 추가해서 FiLM에 통과 
    nov_noise = {}
    for std, Xn in X_train_noise.items():
        Zn = film.transform(Xn, s_dim=sample_obs_dim)  # (N, out_dim)
        nov_noise[std] = compute_novelty(model, Zn, device=DEVICE)

    # compute novelty for circle route 
    circle_route_arrays, circle_route_index_map = [], []
    for fp in circle_route_files:
        x = load_npz_concat_state_action(fp, STATE_KEY, ACTION_KEY)
        circle_route_arrays.append(x)
        circle_route_index_map.extend([(os.path.basename(fp), i) for i in range(x.shape[0])])
        
    X_circle_route_raw = np.concatenate(circle_route_arrays, axis=0) if circle_route_arrays else np.zeros((0, obs_dim), dtype=np.float32)
    X_circle_route = apply_scaler(X_circle_route_raw)
    Z_circle_route = film.transform(X_circle_route, s_dim=sample_obs_dim) if X_circle_route.size else np.zeros((0, obs_dim), np.float32)
    nov_circle_route = compute_novelty(model, Z_circle_route, device=DEVICE) if Z_circle_route.size else np.array([])

    # 8) CSV 저장
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "index", "novelty", "split"])
        for (fname, idx), score in zip(holdout_index_map, nov_holdout):
            writer.writerow([fname, idx, float(score), "validation data(straight route)"])
        for (fname, idx), score in zip(circle_route_index_map, nov_circle_route):
            writer.writerow([fname, idx, float(score), "circle route data"])
    print(f"CSV 저장 완료: {CSV_OUT}")


    # visualize 
    PLOT_OUT1 = "xtrain_with_noise_hist.png"
    PLOT_OUT2 = "novelty_hist.png"
    
    plt.figure()
    plt.hist(X_train.ravel(), bins=80, density=True, alpha=0.6, label="X_train")
    for std, Xn in X_train_noise.items():
        plt.hist(Xn.ravel(), bins=80, density=True, alpha=0.4, label=f"X_train + N(0,{std}^2)")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.title("Distribution: X_train and noise-augmented")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT1, dpi=150)
    plt.close()
    

    plt.figure()
    plt.hist(nov_holdout, bins=80, density=True, alpha=0.6, label="nov_holdout")
    for std, nov in nov_noise.items():
        plt.hist(nov, bins=80, density=True, alpha=0.4, label=f"nov (noise std={std})")
    plt.xlabel("novelty")
    plt.ylabel("density")
    plt.title("Novelty distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT2, dpi=150)
    plt.close()

    print(f"[Saved] {PLOT_OUT2}")

if __name__ == "__main__":
    main()
