"""カウンター候補シーンごとのトラッキング軌道（両チーム分）をキャッシュ用に抽出する。

L_tactic（コンパクトネス・スペース支配）は選手の生軌道に対して直接計算するため、
判定フェーズではモデル学習は不要だが、この軌道データの抽出自体は必要になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kloppy.domain import Team, TrackingDataset

from hs_pinn.counter_events import CounterEvent

GOALKEEPER_POSITION_NAME = "Goalkeeper"


@dataclass
class CounterTrajectory:
    match_id: str
    event_id: str
    team_id: str
    ground: str
    period_id: int
    label: int
    frame_rate: float
    attack_ids: list[str]
    defend_ids: list[str]
    attack_pos: np.ndarray  # (T, n_attack, 2) meters。攻撃方向は常に+x
    defend_pos: np.ndarray  # (T, n_defend, 2) meters
    ball_pos: np.ndarray  # (T, 2) meters


def _outfield_player_ids(team: Team) -> list[str]:
    return [
        p.player_id
        for p in team.players
        if str(p.starting_position) != GOALKEEPER_POSITION_NAME
    ]


def _frame_positions_by_id(frame) -> dict[str, tuple[float, float]]:
    return {
        player.player_id: (data.coordinates.x, data.coordinates.y)
        for player, data in frame.players_data.items()
        if data.coordinates is not None
    }


def build_trajectory(
    tracking: TrackingDataset,
    frames_by_period: dict[int, list],
    counter_event: CounterEvent,
) -> CounterTrajectory | None:
    period_frames = frames_by_period.get(counter_event.period_id)
    if not period_frames:
        return None
    window = period_frames[
        counter_event.start_frame_idx : counter_event.target_frame_idx + 1
    ]
    if len(window) != counter_event.target_frame_idx - counter_event.start_frame_idx + 1:
        return None

    teams = tracking.metadata.teams
    attack_team = next((t for t in teams if t.team_id == counter_event.team_id), None)
    defend_team = next((t for t in teams if t.team_id != counter_event.team_id), None)
    if attack_team is None or defend_team is None:
        return None

    attack_ids = _outfield_player_ids(attack_team)
    defend_ids = _outfield_player_ids(defend_team)
    pitch_length = tracking.metadata.pitch_dimensions.pitch_length
    flip = counter_event.team_ground == "away"

    def to_meters_x(x_normalized: float) -> float:
        x_m = x_normalized * pitch_length
        return pitch_length - x_m if flip else x_m

    def to_meters_y(y_normalized: float, pitch_width: float) -> float:
        y_m = y_normalized * pitch_width
        return pitch_width - y_m if flip else y_m

    pitch_width = tracking.metadata.pitch_dimensions.pitch_width

    T = len(window)
    attack_pos = np.full((T, len(attack_ids), 2), np.nan)
    defend_pos = np.full((T, len(defend_ids), 2), np.nan)
    ball_pos = np.full((T, 2), np.nan)

    for t, frame in enumerate(window):
        positions = _frame_positions_by_id(frame)
        for i, pid in enumerate(attack_ids):
            if pid in positions:
                x, y = positions[pid]
                attack_pos[t, i] = (to_meters_x(x), to_meters_y(y, pitch_width))
        for i, pid in enumerate(defend_ids):
            if pid in positions:
                x, y = positions[pid]
                defend_pos[t, i] = (to_meters_x(x), to_meters_y(y, pitch_width))
        if frame.ball_coordinates is not None:
            ball_pos[t] = (
                to_meters_x(frame.ball_coordinates.x),
                to_meters_y(frame.ball_coordinates.y, pitch_width),
            )

    return CounterTrajectory(
        match_id=counter_event.match_id,
        event_id=counter_event.event_id,
        team_id=counter_event.team_id,
        ground=counter_event.team_ground,
        period_id=counter_event.period_id,
        label=counter_event.label,
        frame_rate=tracking.metadata.frame_rate,
        attack_ids=attack_ids,
        defend_ids=defend_ids,
        attack_pos=attack_pos,
        defend_pos=defend_pos,
        ball_pos=ball_pos,
    )
