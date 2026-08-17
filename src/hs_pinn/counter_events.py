"""カウンター攻撃候補シーンの抽出（判定フェーズ用）。

卒論（ryu622/gnn-counterattack-xai-v2、scientificdata_one6.ipynb）の
「ボール奪取イベント起点＋5秒後到達点での成功/失敗判定」というカウンター定義を、
idsse-dataをkloppyでロードした場合のデータモデルに合わせて再実装したもの。
"""

from __future__ import annotations

from dataclasses import dataclass

from kloppy import sportec
from kloppy.domain import EventDataset, EventType, Orientation, TrackingDataset

# DFL生データの 'TacklingGame' / 'BallDeflection' は、kloppyでは EventType.GENERIC に
# 分類されるが、元のイベント名は event_name に保持される。
# 'BallClaiming' は kloppy が直接 EventType.RECOVERY にマッピングする。
_GENERIC_RECOVERY_NAMES = {"TacklingGame", "BallDeflection"}

OBSERVATION_WINDOW_FRAMES = 25  # 1秒 @ 25fps（モデル入力用の観測ウィンドウ）
PREDICTION_HORIZON_FRAMES = 125  # 5秒 @ 25fps（成功/失敗判定のホライズン）
DEEP_AREA_THRESHOLD_M = 25.0  # ピッチ中央から敵陣方向への閾値（m）
PROGRESS_THRESHOLD_M = 5.0  # 前進とみなす最小距離（m）


@dataclass
class CounterEvent:
    match_id: str
    event_id: str
    team_id: str
    team_ground: str  # "home" or "away"
    period_id: int
    start_frame_idx: int  # 該当ピリオド内のフレーム列インデックス
    target_frame_idx: int
    start_progress_m: float
    target_progress_m: float
    label: int  # 1 = 成功, 0 = 失敗


def is_recovery_event(event) -> bool:
    """ボールの支配権が実際に相手側へ移った瞬間かどうかを判定する。

    EventType.RECOVERY（BallClaiming由来）はkloppyが「相手ボールを奪った」
    ケースとしてのみ生成するため無条件に真。EventType.GENERIC
    （TacklingGame/BallDeflection由来）はデュエルの勝敗を表すだけで、
    既にボールを保持していた側が競り勝った場合（ターンオーバーなし）も
    含まれてしまうため、raw_eventの`PossessionChange == "true"`
    （このデュエルでボール保持チームが入れ替わった）を追加で要求する。
    """
    if event.event_type == EventType.RECOVERY:
        return True
    if event.event_type != EventType.GENERIC or event.event_name not in _GENERIC_RECOVERY_NAMES:
        return False
    raw = event.raw_event or {}
    return raw.get("PossessionChange") == "true"


def _recovering_team_id(event) -> str | None:
    """奪取した側のteam_idを解決する。

    kloppyはEventType.RECOVERY（BallClaiming由来）には`event.team`を
    セットするが、EventType.GENERIC（TacklingGame/BallDeflection由来）は
    DFL生データに単一の`Team`属性がないためNoneのままになる。
    後者はraw_eventの`WinnerTeam`（奪取側）を直接参照する。
    """
    if event.team is not None:
        return event.team.team_id
    raw = event.raw_event or {}
    return raw.get("WinnerTeam")


def load_match(match_id: str) -> tuple[TrackingDataset, EventDataset]:
    """トラッキング・イベントデータをロードし、home/awayが常に一定方向を
    攻撃する向きに正規化する（卒論の period依存flipロジックに相当）。"""
    tracking = sportec.load_open_tracking_data(match_id=match_id)
    events = sportec.load_open_event_data(match_id=match_id)
    tracking = tracking.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
    events = events.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
    return tracking, events


def _frames_by_period(tracking: TrackingDataset) -> dict[int, list]:
    by_period: dict[int, list] = {}
    for frame in tracking.records:
        by_period.setdefault(frame.period.id, []).append(frame)
    return by_period


def _frame_index(timestamp, frame_rate: float) -> int:
    return round(timestamp.total_seconds() * frame_rate)


def _progress_m(x_normalized: float, pitch_length: float, ground: str) -> float:
    """攻撃側ゴール方向への前進距離（m）。STATIC_HOME_AWAY正規化後は
    homeが常に+x方向、awayが常に-x方向に攻撃する前提。"""
    x_m = x_normalized * pitch_length
    return x_m if ground == "home" else pitch_length - x_m


def extract_counter_events(
    match_id: str,
    tracking: TrackingDataset | None = None,
    events: EventDataset | None = None,
) -> list[CounterEvent]:
    if tracking is None or events is None:
        tracking, events = load_match(match_id)
    pitch_length = tracking.metadata.pitch_dimensions.pitch_length
    frame_rate = tracking.metadata.frame_rate
    frames_by_period = _frames_by_period(tracking)
    ground_by_team_id = {
        team.team_id: team.ground.value for team in tracking.metadata.teams
    }

    counter_events: list[CounterEvent] = []

    for event in events.records:
        if not is_recovery_event(event):
            continue
        team_id = _recovering_team_id(event)
        ground = ground_by_team_id.get(team_id)
        if ground not in ("home", "away"):
            continue

        period_frames = frames_by_period.get(event.period.id)
        if not period_frames:
            continue

        start_idx = _frame_index(event.timestamp, frame_rate)
        target_idx = start_idx + PREDICTION_HORIZON_FRAMES

        if start_idx < 0 or start_idx >= len(period_frames):
            continue
        if target_idx >= len(period_frames):
            continue  # ピリオド終了までの残り時間が足りない（卒論と同じガード条件）

        start_frame = period_frames[start_idx]
        target_frame = period_frames[target_idx]
        if start_frame.ball_coordinates is None or target_frame.ball_coordinates is None:
            continue

        start_progress = _progress_m(start_frame.ball_coordinates.x, pitch_length, ground)
        target_progress = _progress_m(target_frame.ball_coordinates.x, pitch_length, ground)

        is_in_deep_area = target_progress > (pitch_length / 2 + DEEP_AREA_THRESHOLD_M)
        is_progressing = (target_progress - start_progress) > PROGRESS_THRESHOLD_M
        label = 1 if (is_in_deep_area and is_progressing) else 0

        counter_events.append(
            CounterEvent(
                match_id=match_id,
                event_id=event.event_id,
                team_id=team_id,
                team_ground=ground,
                period_id=event.period.id,
                start_frame_idx=start_idx,
                target_frame_idx=target_idx,
                start_progress_m=start_progress,
                target_progress_m=target_progress,
                label=label,
            )
        )

    return counter_events
