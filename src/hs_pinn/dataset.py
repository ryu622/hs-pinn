"""カウンター候補シーンの軌道データ（`CounterTrajectory`）を、モデル学習用の
テンソルに変換する（計画書3.2節の入出力形状に対応）。

- 入力：観測窓（奪取後1秒 = 25フレーム）の攻撃側・守備側・ボールの位置/速度
- 予測対象：残り（奪取後1秒〜5秒 = 100フレーム）の攻撃側選手の位置

選手数はイベントごとに変動する（オンピッチ9〜10人程度）ため、固定長
（`MAX_ATTACK_PLAYERS` / `MAX_DEFEND_PLAYERS`）へゼロパディングし、
有効な選手を示すbool maskを併せて返す（Self-Attentionのpadding maskに使う）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from hs_pinn.counter_events import OBSERVATION_WINDOW_FRAMES, PREDICTION_HORIZON_FRAMES
from hs_pinn.space_control import estimate_velocities
from hs_pinn.trajectories import CounterTrajectory

MAX_ATTACK_PLAYERS = 10
MAX_DEFEND_PLAYERS = 10
INPUT_FRAMES = OBSERVATION_WINDOW_FRAMES  # 25 (奪取後1秒)
TARGET_FRAMES = PREDICTION_HORIZON_FRAMES + 1 - INPUT_FRAMES  # 101 (残り約4秒)


@dataclass
class TrainingSample:
    match_id: str
    event_id: str
    label: int

    # 入力（観測窓、shape=(INPUT_FRAMES, max_players, 2)）
    input_attack_pos: torch.Tensor
    input_attack_vel: torch.Tensor
    input_defend_pos: torch.Tensor
    input_defend_vel: torch.Tensor
    input_ball_pos: torch.Tensor
    input_ball_vel: torch.Tensor
    attack_mask: torch.Tensor  # (max_attack,) bool
    defend_mask: torch.Tensor  # (max_defend,) bool

    # ハード制約層への初期状態（観測窓の最終フレーム）。従来は攻撃側予測のみだったが、
    # 計画書10節（守備側予測への転換）に伴い守備側の初期状態も追加した。
    init_position: torch.Tensor  # (max_attack, 2) 旧来通り攻撃側
    init_velocity: torch.Tensor  # (max_attack, 2)
    init_defend_position: torch.Tensor  # (max_defend, 2)
    init_defend_velocity: torch.Tensor  # (max_defend, 2)

    # 予測対象（shape=(T_target, max_attack, 2)）
    target_attack_pos: torch.Tensor

    # 予測対象区間における守備側の実軌道（正解データ）。10節以前は守備側を予測対象に
    # 含めない設計だったため固定コンテキスト用に保持していたが、10節では守備側予測の
    # 学習ターゲットとしても使う。
    target_defend_pos: torch.Tensor  # (T_target, max_defend, 2)


def _active_indices(pos: np.ndarray) -> np.ndarray:
    """全フレームでNaNでない（＝ウィンドウ全体を通じてオンピッチにいた）選手のインデックス。"""
    valid_mask = ~np.isnan(pos[:, :, 0]).any(axis=0)
    return np.where(valid_mask)[0]


def _pad(arr: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, n, 2) -> (T, max_n, 2) にゼロパディングし、有効選手を示すbool mask(max_n,)を返す。"""
    T, n, _ = arr.shape
    out = np.zeros((T, max_n, 2), dtype=np.float32)
    mask = np.zeros(max_n, dtype=bool)
    take = min(n, max_n)
    out[:, :take] = arr[:, :take]
    mask[:take] = True
    return out, mask


def build_sample(
    traj: CounterTrajectory,
    max_attack: int = MAX_ATTACK_PLAYERS,
    max_defend: int = MAX_DEFEND_PLAYERS,
    input_frames: int = INPUT_FRAMES,
) -> TrainingSample | None:
    a_idx = _active_indices(traj.attack_pos)
    d_idx = _active_indices(traj.defend_pos)
    if len(a_idx) == 0 or len(d_idx) == 0:
        return None

    attack_pos_active = traj.attack_pos[:, a_idx]
    defend_pos_active = traj.defend_pos[:, d_idx]
    ball_pos_active = traj.ball_pos[:, None, :]

    attack_vel_active = estimate_velocities(attack_pos_active, traj.frame_rate)
    defend_vel_active = estimate_velocities(defend_pos_active, traj.frame_rate)
    ball_vel_active = estimate_velocities(ball_pos_active, traj.frame_rate)

    attack_pos, attack_mask = _pad(attack_pos_active, max_attack)
    attack_vel, _ = _pad(attack_vel_active, max_attack)
    defend_pos, defend_mask = _pad(defend_pos_active, max_defend)
    defend_vel, _ = _pad(defend_vel_active, max_defend)
    ball_pos, _ = _pad(ball_pos_active, 1)
    ball_vel, _ = _pad(ball_vel_active, 1)

    T = attack_pos.shape[0]

    return TrainingSample(
        match_id=traj.match_id,
        event_id=traj.event_id,
        label=traj.label,
        input_attack_pos=torch.from_numpy(attack_pos[:input_frames]),
        input_attack_vel=torch.from_numpy(attack_vel[:input_frames]),
        input_defend_pos=torch.from_numpy(defend_pos[:input_frames]),
        input_defend_vel=torch.from_numpy(defend_vel[:input_frames]),
        input_ball_pos=torch.from_numpy(ball_pos[:input_frames]),
        input_ball_vel=torch.from_numpy(ball_vel[:input_frames]),
        attack_mask=torch.from_numpy(attack_mask),
        defend_mask=torch.from_numpy(defend_mask),
        init_position=torch.from_numpy(attack_pos[input_frames - 1]),
        init_velocity=torch.from_numpy(attack_vel[input_frames - 1]),
        init_defend_position=torch.from_numpy(defend_pos[input_frames - 1]),
        init_defend_velocity=torch.from_numpy(defend_vel[input_frames - 1]),
        target_attack_pos=torch.from_numpy(attack_pos[input_frames:T]),
        target_defend_pos=torch.from_numpy(defend_pos[input_frames:T]),
    )


class CounterAttackDataset(Dataset):
    def __init__(self, trajectories: list[CounterTrajectory]) -> None:
        samples = [build_sample(t) for t in trajectories]
        self.samples = [s for s in samples if s is not None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.samples[idx]


def collate_samples(samples: list[TrainingSample]) -> dict:
    """DataLoaderのデフォルトcollateは`TrainingSample`（dataclass）を扱えないため、
    フィールドごとにstackした辞書を返す。"""
    return {
        "match_id": [s.match_id for s in samples],
        "event_id": [s.event_id for s in samples],
        "label": torch.tensor([s.label for s in samples], dtype=torch.long),
        "input_attack_pos": torch.stack([s.input_attack_pos for s in samples]),
        "input_attack_vel": torch.stack([s.input_attack_vel for s in samples]),
        "input_defend_pos": torch.stack([s.input_defend_pos for s in samples]),
        "input_defend_vel": torch.stack([s.input_defend_vel for s in samples]),
        "input_ball_pos": torch.stack([s.input_ball_pos for s in samples]),
        "input_ball_vel": torch.stack([s.input_ball_vel for s in samples]),
        "attack_mask": torch.stack([s.attack_mask for s in samples]),
        "defend_mask": torch.stack([s.defend_mask for s in samples]),
        "init_position": torch.stack([s.init_position for s in samples]),
        "init_velocity": torch.stack([s.init_velocity for s in samples]),
        "init_defend_position": torch.stack([s.init_defend_position for s in samples]),
        "init_defend_velocity": torch.stack([s.init_defend_velocity for s in samples]),
        "target_attack_pos": torch.stack([s.target_attack_pos for s in samples]),
        "target_defend_pos": torch.stack([s.target_defend_pos for s in samples]),
    }


# 試合単位での分割（データリーク防止）。7試合中5/1/1で暫定的に分割。
# 小規模データ(502件)のため、split間で分布が偏る可能性がある点は要留意（documents/phase0_report.md参照）。
MATCH_SPLIT = {
    "train": ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY"],
    "valid": ["J03WQQ"],
    "test": ["J03WR9"],
}


def split_trajectories(
    trajectories: list[CounterTrajectory],
) -> dict[str, list[CounterTrajectory]]:
    out: dict[str, list[CounterTrajectory]] = {"train": [], "valid": [], "test": []}
    for t in trajectories:
        for split, match_ids in MATCH_SPLIT.items():
            if t.match_id in match_ids:
                out[split].append(t)
                break
    return out
