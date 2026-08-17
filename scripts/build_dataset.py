"""idsse-dataの全7試合からカウンター候補シーン＋軌道データを抽出し、
data/processed/counter_trajectories.pkl にキャッシュする。

判定フェーズ（L_tacticの定式化・分布確認）は毎回kloppyでロードし直すと
1試合あたり約40秒かかるため、このキャッシュを介して繰り返し利用する。

実行: uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from hs_pinn.counter_events import extract_counter_events, load_match
from hs_pinn.trajectories import build_trajectory

MATCH_IDS = [
    "J03WPY",
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WQQ",
    "J03WR9",
]

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"


def _frames_by_period(tracking):
    by_period: dict[int, list] = {}
    for frame in tracking.records:
        by_period.setdefault(frame.period.id, []).append(frame)
    return by_period


def main() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_trajectories = []

    for match_id in MATCH_IDS:
        t0 = time.time()
        tracking, events = load_match(match_id)
        counter_events = extract_counter_events(match_id, tracking=tracking, events=events)
        frames_by_period = _frames_by_period(tracking)

        match_trajectories = []
        for ce in counter_events:
            traj = build_trajectory(tracking, frames_by_period, ce)
            if traj is not None:
                match_trajectories.append(traj)

        elapsed = time.time() - t0
        print(
            f"{match_id}: {len(counter_events)} events -> "
            f"{len(match_trajectories)} trajectories ({elapsed:.1f}s)"
        )
        all_trajectories.extend(match_trajectories)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_trajectories, f)

    print(f"\nTotal: {len(all_trajectories)} trajectories saved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
