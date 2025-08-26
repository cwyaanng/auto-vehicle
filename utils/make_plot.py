
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# 설정값
# ---------------------------
CSV1 = "result/total_reward_episode_sacrnd.csv"
CSV2 = "result/total_reward_episode_pure_sac_512.csv"
CSV3 = "result/total_reward_episode_sac_offlinetoOnline.csv"

LABEL1 = "SAC with Q value adjustment with RND"
LABEL2 = "Pure SAC"
LABEL3 = "SAC Offline to Online"

SMOOTH_WINDOW = 100    # 이동평균 윈도우(샘플 개수)
X_GAMMA = 2.0         # x축 감마(>1이면 후반부 확장)
LINEWIDTH = 3.0       # 선 두께
Y_LINTHRESH = 10.0    # symlog에서 선형 구간 절대값 임계치 (±10 구간 압축)
Y_BASE = 10.0         # 로그 밑 (10진 로그)
OUTFILE = f"plots/total_reward_episode_symlog_smooth_percent_gamma{X_GAMMA}.png"

TITLE = "Total reward of episode"
X_LABEL = "Progress (%)"
Y_LABEL = "Total reward of episode"

# ---------------------------
# 유틸
# ---------------------------
def moving_average(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()

def to_percent(step_values):
    x = np.asarray(step_values, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 100.0

def warp_percent(percent, gamma=2.0):
    p = np.asarray(percent, dtype=float)
    p = np.clip(p, 0.0, 100.0)
    return (p / 100.0) ** gamma * 100.0

def load_prepare(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path)
    if "Step" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"{path}: 'Step' 또는 'Value' 컬럼 누락")
    df = df[["Step", "Value"]].dropna()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna().sort_values("Step")
    df = df.drop_duplicates(subset="Step", keep="last").reset_index(drop=True)
    df["Smoothed"] = moving_average(df["Value"], window=SMOOTH_WINDOW)
    df["Percent"] = to_percent(df["Step"])
    return df

# ---------------------------
# 데이터 로드
# ---------------------------
df1 = load_prepare(CSV1)
df2 = load_prepare(CSV2)
df3 = load_prepare(CSV3)

# ---------------------------
# 플롯 (y: symlog + x: 감마 워프된 0~100%)
# ---------------------------
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(12, 8))

for df, label in [(df1, LABEL1), (df2, LABEL2), (df3, LABEL3)]:
    x = warp_percent(df["Percent"].to_numpy(), gamma=X_GAMMA)
    y = df["Smoothed"].to_numpy()  # symlog는 음수/0도 처리 가능(±Y_LINTHRESH 구간은 선형)
    ax.plot(x, y, label=label, linewidth=LINEWIDTH)

# y축: 대칭 로그 스케일 (±Y_LINTHRESH 선형 영역, 바깥쪽 로그)
ax.set_yscale("symlog", linthresh=Y_LINTHRESH, base=Y_BASE)
ax.set_ylim(-1e3, None) 
ax.set_title(TITLE, fontsize=30)
ax.set_xlabel(X_LABEL, fontsize=30)
ax.set_ylabel(Y_LABEL, fontsize=30)
ax.grid(True, which="both")
ax.legend(fontsize=15)

# x축 눈금을 0,25,50,75,100%로 고정하되, 워프된 위치에 배치
ticks_pct = np.array([0, 25, 50, 75, 100], dtype=float)
tick_pos = warp_percent(ticks_pct, gamma=X_GAMMA)
ax.set_xticks(tick_pos)
ax.set_xticklabels([f"{int(t)}%" for t in ticks_pct])

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUTFILE}")


# make_plot_waypoint.py
# waypoint ahead reward 시각화
# (symlog y축, 0~100% 진행도, 감마 워프, smoothing, linewidth 굵게, y축 하한 1e3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# 설정값
# ---------------------------
CSV1 = "result/waypoint_ahead_episode_sacrnd.csv"
CSV2 = "result/waypoint_ahead_pure_sac_512.csv"
CSV3 = "result/waypoint_ahead_episode_sac_offlinetoOnline.csv"

LABEL1 = "SAC with Q value adjustment with RND"
LABEL2 = "Pure SAC"
LABEL3 = "SAC Offline to Online"

SMOOTH_WINDOW = 75  # 이동평균 윈도우
X_GAMMA = 2.0         # x축 감마
LINEWIDTH = 3.0       # 선 두께
Y_LINTHRESH = 10.0    # symlog 선형 구간
Y_BASE = 10.0         # 로그 밑
OUTFILE = f"plots/waypoint_ahead_symlog_smooth_percent_gamma{X_GAMMA}.png"

TITLE = "Reached waypoint at the end of the episode"
X_LABEL = "Progress (%)"
Y_LABEL = "Reached waypoint"

# ---------------------------
# 유틸
# ---------------------------
def moving_average(series, window=100):
    return series.rolling(window=window, min_periods=1).mean()

def to_percent(step_values):
    x = np.asarray(step_values, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 100.0

def warp_percent(percent, gamma=2.0):
    p = np.asarray(percent, dtype=float)
    p = np.clip(p, 0.0, 100.0)
    return (p / 100.0) ** gamma * 100.0

def load_prepare(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_csv(path)
    if "Step" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"{path}: 'Step' 또는 'Value' 컬럼 누락")
    df = df[["Step", "Value"]].dropna()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna().sort_values("Step")
    df = df.drop_duplicates(subset="Step", keep="last").reset_index(drop=True)
    df["Smoothed"] = moving_average(df["Value"], window=SMOOTH_WINDOW)
    df["Percent"] = to_percent(df["Step"])
    return df

# ---------------------------
# 공통 스케일(글로벌) 설정
# ---------------------------
GLOBAL_MIN = 0.0
GLOBAL_MAX = 1001.0

def normalize_global(arr, lo=GLOBAL_MIN, hi=GLOBAL_MAX):
    arr = np.asarray(arr, dtype=float)
    denom = hi - lo
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / denom, 0.0, 1.0)

# ---------------------------
# 데이터 로드
# ---------------------------
df1 = load_prepare(CSV1)
df2 = load_prepare(CSV2)
df3 = load_prepare(CSV3)

# ---------------------------
# 플롯
# ---------------------------
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(12, 8))

for df, label in [(df1, LABEL1), (df2, LABEL2), (df3, LABEL3)]:
    x = warp_percent(df["Percent"].to_numpy(), gamma=X_GAMMA)
    y = normalize_global(df["Smoothed"])  # ★ per-curve 아님, 공통 스케일
    ax.plot(x, y, label=label, linewidth=LINEWIDTH)

# y축: 0~1 스케일 + y=1 기준선
ax.set_yscale("linear")
ax.set_ylim(0, 1.0)
ax.axhline(1.0, color="red", linestyle="--", linewidth=2, label=f"Upper bound")

ax.set_title(TITLE, fontsize=30)
ax.set_xlabel(X_LABEL, fontsize=30)
ax.set_ylabel("Reached waypoint", fontsize=30)
ax.grid(True, which="both")
ax.legend(fontsize=15)

# x축 눈금 유지
ticks_pct = np.array([0, 25, 50, 75, 100], dtype=float)
tick_pos = warp_percent(ticks_pct, gamma=X_GAMMA)
ax.set_xticks(tick_pos)
ax.set_xticklabels([f"{int(t)}%" for t in ticks_pct])

plt.tight_layout()
plt.savefig("plots/waypoint_ahead_norm01_global.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: plots/waypoint_ahead_norm01_global.png")
