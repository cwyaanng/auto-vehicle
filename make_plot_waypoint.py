# make_plot_waypoint.py
# waypoint ahead reward 시각화 (3개 비교, 선만 표시, y축 상한 1.1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# 설정값
# ---------------------------
CSV1 = "result/final_result/wapoint_ahead_제안기법_1차.csv"
CSV2 = "result/final_result/waypoint_ahead_제안기법_retest.csv"    # 경로 확인!
CSV3 = "result/final_result/waypoint_ahead_3차실험.csv"

LABEL1 = "TEST1"
LABEL2 = "TEST2"
LABEL3 = "TEST3"

SMOOTH_WINDOW = 100    # 이동평균 윈도우
X_GAMMA = 2.0          # x축 감마
LINEWIDTH = 3.0        # 선 두께
OUTFILE = "plots/0929이후plot/comparison_waypoint.png"

TITLE = "Reached waypoint at the end of the episode"
X_LABEL = "Progress (%)"
Y_LABEL = "Reached waypoint (normalized)"

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
    path = Path(path)
    if not path.exists():
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

# 공통 스케일(글로벌) 설정: 원 자료의 범위를 [lo, hi]로 보고 0~1로 정규화
GLOBAL_MIN = 0.0
GLOBAL_MAX = 1001.0
def normalize_global(arr, lo=GLOBAL_MIN, hi=GLOBAL_MAX):
    arr = np.asarray(arr, dtype=float)
    denom = hi - lo
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / denom, 0.0, 1.0)

# ---------------------------
# 비교 대상 정의 (여기만 수정하면 N개로 확장 가능)
# ---------------------------
SERIES = [
    (CSV1, LABEL1),
    (CSV2, LABEL2),
    (CSV3, LABEL3),
]

# ---------------------------
# 데이터 로드
# ---------------------------
loaded = []
errors = []
for csv_path, label in SERIES:
    try:
        df = load_prepare(csv_path)
        loaded.append((df, label))
    except Exception as e:
        errors.append(f"{label}: {e}")

if not loaded:
    raise RuntimeError("로딩된 데이터가 없습니다.\n" + "\n".join(errors))

# ---------------------------
# 플롯
# ---------------------------
plt.close("all")  # 이전 그림/스타일 잔상 제거
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(12, 8))

for df, label in loaded:
    x = warp_percent(df["Percent"].to_numpy(), gamma=X_GAMMA)
    y = normalize_global(df["Smoothed"])  # 공통 스케일(0~1)
    ax.plot(x, y, label=label, linewidth=LINEWIDTH, linestyle="-", marker=None)

# y축: 0~1.1 (상한 1.1로 여유)
ax.set_yscale("linear")
ax.set_ylim(0, 1.1)
ax.axhline(1.0, linestyle="--", linewidth=2, label="Upper bound (1.0)")

# 제목/라벨/격자/범례
ax.set_title(TITLE, fontsize=30)
ax.set_xlabel(X_LABEL, fontsize=30)
ax.set_ylabel(Y_LABEL, fontsize=30)
ax.grid(True, which="both")
ax.legend(fontsize=15)  # 범례 박스 네모까지 없애고 싶으면 frameon=False

# x축: 0,25,50,75,100% 눈금을 워프된 위치에 표시
ticks_pct = np.array([0, 25, 50, 75, 100], dtype=float)
tick_pos = warp_percent(ticks_pct, gamma=X_GAMMA)
ax.set_xticks(tick_pos)
ax.set_xticklabels([f"{int(t)}%" for t in ticks_pct])

# 저장
Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUTFILE}")
if errors:
    print("경고(로딩 실패 항목):")
    for msg in errors:
        print(" -", msg)
