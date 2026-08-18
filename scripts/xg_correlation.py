"""xGとの相関（計画書7節「既存指標との相関」）。

DFL生データのシュートイベントには公式のxG値が含まれており、自前でxGモデルを
構築する必要はない。各カウンター候補シーン（奪取イベント）から一定時間内に、
同じチーム・同じピリオドでのシュートが記録されていれば、そのxGを
「生み出した決定機の質」という連続的な結果指標として扱い、L_tactic
（コンパクトネス・スペース支配）との相関を確認する。

判定②（成功/失敗の二値）を補完する、より粒度の細かい行動結果指標。

実行: uv run python scripts/xg_correlation.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import stats

from hs_pinn.counter_events import extract_counter_events, load_match
from hs_pinn.space_control import build_pitch_grid, voronoi_dominance
from hs_pinn.tactic_metrics import compactness_scalar

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "counter_trajectories.pkl"
SHOT_WINDOW_S = 10.0  # 奪取からこの秒数以内のシュートを「このカウンターの結果」とみなす
FRAME_STRIDE = 10
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def extract_shots(events) -> list[dict]:
    shots = []
    for e in events.records:
        if str(e.event_type) != "EventType.SHOT":
            continue
        raw = e.raw_event or {}
        xg = raw.get("xG")
        if xg is None or e.team is None:
            continue
        shots.append(
            {
                "team_id": e.team.team_id,
                "period_id": e.period.id,
                "timestamp_s": e.timestamp.total_seconds(),
                "xg": float(xg),
            }
        )
    return shots


def space_dominance_scalar(traj, grid) -> float:
    T = traj.attack_pos.shape[0]
    values = [
        voronoi_dominance(grid, traj.attack_pos[t], traj.defend_pos[t])
        for t in range(0, T, FRAME_STRIDE)
    ]
    return float(np.nanmean(values))


def main() -> None:
    with open(CACHE_PATH, "rb") as f:
        trajs = pickle.load(f)
    traj_by_key = {(t.match_id, t.event_id): t for t in trajs}
    grid = build_pitch_grid(PITCH_LENGTH, PITCH_WIDTH)

    records = []
    n_with_shot = 0

    for match_id in MATCH_IDS:
        tracking, events = load_match(match_id)
        counter_events = extract_counter_events(match_id, tracking=tracking, events=events)
        shots = extract_shots(events)
        frame_rate = tracking.metadata.frame_rate

        for ce in counter_events:
            key = (match_id, ce.event_id)
            traj = traj_by_key.get(key)
            if traj is None:
                continue

            recovery_t = ce.start_frame_idx / frame_rate
            candidate_xgs = [
                s["xg"]
                for s in shots
                if s["team_id"] == ce.team_id
                and s["period_id"] == ce.period_id
                and 0 <= (s["timestamp_s"] - recovery_t) <= SHOT_WINDOW_S
            ]
            xg = max(candidate_xgs) if candidate_xgs else np.nan
            if candidate_xgs:
                n_with_shot += 1

            compact = compactness_scalar(traj.attack_pos, "longitudinal_variance")
            space = space_dominance_scalar(traj, grid)

            records.append(
                {
                    "match_id": match_id,
                    "event_id": ce.event_id,
                    "label": ce.label,
                    "xg": xg,
                    "compact": compact,
                    "space": space,
                }
            )

        print(f"{match_id}: {len(counter_events)} events processed")

    n_total = len(records)
    print(f"\nn_total={n_total}, n_with_shot_within_{SHOT_WINDOW_S}s={n_with_shot} "
          f"({100*n_with_shot/n_total:.1f}%)")

    xg = np.array([r["xg"] for r in records])
    label = np.array([r["label"] for r in records])
    compact = np.array([r["compact"] for r in records])
    space = np.array([r["space"] for r in records])

    has_shot = ~np.isnan(xg)
    print(f"\n=== シュートありサブセット (n={has_shot.sum()}) ===")
    r_label, p_label = stats.pointbiserialr(label[has_shot], xg[has_shot])
    print(f"xG vs success label: r={r_label:+.3f} p={p_label:.4f} (整合性チェック：シュート機会があれば成功ラベルとは一致するはず)")

    valid_c = has_shot & ~np.isnan(compact)
    r_c, p_c = stats.pearsonr(compact[valid_c], xg[valid_c])
    print(f"xG vs L_compact(longitudinal_variance): r={r_c:+.3f} p={p_c:.4f} (n={valid_c.sum()})")

    valid_s = has_shot & ~np.isnan(space)
    r_s, p_s = stats.pearsonr(space[valid_s], xg[valid_s])
    print(f"xG vs L_space(voronoi_dominance): r={r_s:+.3f} p={p_s:.4f} (n={valid_s.sum()})")

    out_path = CACHE_PATH.parent / "xg_correlation.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"\nsaved records to {out_path}")


if __name__ == "__main__":
    main()
