# make_plot_reward_scaled.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# 설정값
# ---------------------------
CSV1 = "result/final_result/episode_return_제안기법_1차.csv"
CSV2 = "result/final_result/episode_reward_제안기법_retest.csv"
CSV3 = "result/final_result/episode_reward_3차실험.csv"  # 필요 시 다른 경로로 교체

LABEL1 = "TEST1"
LABEL2 = "TEST2"   # 새 라벨 추가
LABEL3 = "TEST3"

SMOOTH_WINDOW = 100     # 이동평균 윈도우(샘플 개수)
X_GAMMA = 2.0           # x축 감마(>1: 후반부 확대)
LINEWIDTH = 3.0         # 선 두께
OUTFILE = "plots/0929이후plot/comparison_reward.png"

TITLE = "Total reward of episode"
X_LABEL = "Progress (%)"
Y_LABEL = "Total reward of episode"

# === Y 스케일 옵션 ===
Y_SCALE = "linear"      # "linear" 또는 "symlog"
AUTO_YLIM = True        # 분위수 기반 자동 확대
Y_QLOW = 0.05           # 하위 5% 분위수
Y_QHIGH = 0.95          # 상위 95% 분위수
Y_MIN = None            # 수동 하한 (예: 200). None이면 자동/분위수 사용
Y_MAX = None            # 수동 상한
MARGIN_RATIO = 0.05     # 자동 확대 시 상하단 여백 비율

# symlog 세부 설정( Y_SCALE == "symlog"일 때 사용 )
Y_LINTHRESH = 10.0      # 선형 영역 임계치(작게 할수록 로그 압축 범위↑)
Y_BASE = 10.0           # 로그 밑

# ---------------------------
# 비교 대상 정의 (여기만 수정하면 N개로 확장 가능)
# ---------------------------
SERIES = [
    (CSV1, LABEL1),
    (CSV2, LABEL2),
    (CSV3, LABEL3),
]

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
        errors.append(str(e))

if not loaded:
    raise RuntimeError("로딩된 데이터가 없습니다.\n" + "\n".join(errors))

# y-스케일 자동 결정을 위한 전체 y 배열
y_all_parts = [df["Smoothed"].to_numpy() for df, _ in loaded]
y_all = np.concatenate(y_all_parts)
y_all = y_all[np.isfinite(y_all)]

# ---------------------------
# 플롯 (y 스케일 토글 + 분위수 확대 지원)
# ---------------------------
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(12, 8))

for df, label in loaded:
    x = warp_percent(df["Percent"].to_numpy(), gamma=X_GAMMA)
    y = df["Smoothed"].to_numpy()
    ax.plot(x, y, label=label, linewidth=LINEWIDTH)

# y축 스케일 적용
if Y_SCALE == "symlog":
    ax.set_yscale("symlog", linthresh=Y_LINTHRESH, base=Y_BASE)
else:
    ax.set_yscale("linear")

# y축 범위 결정 (직접 지정 우선, 없으면 분위수 기반 자동 확대)
if (Y_MIN is not None) or (Y_MAX is not None):
    ymin = np.nanmin(y_all) if Y_MIN is None else Y_MIN
    ymax = np.nanmax(y_all) if Y_MAX is None else Y_MAX
else:
    if AUTO_YLIM and y_all.size > 0:
        qlow = np.nanquantile(y_all, Y_QLOW)
        qhigh = np.nanquantile(y_all, Y_QHIGH)
        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
            qlow, qhigh = np.nanmin(y_all), np.nanmax(y_all)
        margin = (qhigh - qlow) * MARGIN_RATIO
        ymin = qlow - margin
        ymax = qhigh + margin
    else:
        ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)

ax.set_ylim(ymin, ymax)

# 제목/라벨/격자/범례
ax.set_title(TITLE, fontsize=30)
ax.set_xlabel(X_LABEL, fontsize=30)
ax.set_ylabel(Y_LABEL, fontsize=30)
ax.grid(True, which="both")
ax.legend(fontsize=15)

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
