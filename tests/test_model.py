import torch

from hs_pinn.dataset import MAX_ATTACK_PLAYERS, MAX_DEFEND_PLAYERS, TARGET_FRAMES
from hs_pinn.hard_constraints import PitchBounds
from hs_pinn.model import TrajectoryBackbone

INPUT_FRAMES = 25


def make_synthetic_batch(batch_size: int = 4, n_valid_attack: int = 8, n_valid_defend: int = 9) -> dict:
    """kloppy/実データに依存しない、形状だけ本物と一致する合成バッチ。"""
    g = torch.Generator().manual_seed(0)

    def rand_pos(n):
        return torch.rand(batch_size, INPUT_FRAMES, n, 2, generator=g) * 50

    def rand_vel(n):
        return torch.randn(batch_size, INPUT_FRAMES, n, 2, generator=g)

    attack_mask = torch.zeros(batch_size, MAX_ATTACK_PLAYERS, dtype=torch.bool)
    attack_mask[:, :n_valid_attack] = True
    defend_mask = torch.zeros(batch_size, MAX_DEFEND_PLAYERS, dtype=torch.bool)
    defend_mask[:, :n_valid_defend] = True

    return {
        "input_attack_pos": rand_pos(MAX_ATTACK_PLAYERS),
        "input_attack_vel": rand_vel(MAX_ATTACK_PLAYERS),
        "input_defend_pos": rand_pos(MAX_DEFEND_PLAYERS),
        "input_defend_vel": rand_vel(MAX_DEFEND_PLAYERS),
        "input_ball_pos": rand_pos(1),
        "input_ball_vel": rand_vel(1),
        "attack_mask": attack_mask,
        "defend_mask": defend_mask,
        "init_position": torch.rand(batch_size, MAX_ATTACK_PLAYERS, 2, generator=g) * 50,
        "init_velocity": torch.randn(batch_size, MAX_ATTACK_PLAYERS, 2, generator=g),
        "target_attack_pos": torch.rand(batch_size, TARGET_FRAMES, MAX_ATTACK_PLAYERS, 2, generator=g) * 50,
    }


def make_model() -> TrajectoryBackbone:
    return TrajectoryBackbone(
        embed_dim=8, hidden_dim=16, n_heads=2, pitch_bounds=PitchBounds(0, 105, 0, 68)
    )


def test_output_shapes():
    model = make_model()
    batch = make_synthetic_batch()
    pos, vel, acc = model(batch)

    expected = (4, MAX_ATTACK_PLAYERS, TARGET_FRAMES, 2)
    assert pos.shape == expected
    assert vel.shape == expected
    assert acc.shape == expected


def test_output_respects_hard_constraints():
    model = make_model()
    batch = make_synthetic_batch()
    pos, vel, acc = model(batch)

    assert torch.all(acc.abs() <= model.hard_constraints.a_max + 1e-4)
    assert torch.all(vel.abs() <= model.hard_constraints.v_max + 1e-4)
    pb = model.hard_constraints.pitch_bounds
    assert torch.all(pos[..., 0] >= pb.x_min - 1e-4)
    assert torch.all(pos[..., 0] <= pb.x_max + 1e-4)
    assert torch.all(pos[..., 1] >= pb.y_min - 1e-4)
    assert torch.all(pos[..., 1] <= pb.y_max + 1e-4)


def test_gradients_flow_end_to_end():
    model = make_model()
    batch = make_synthetic_batch()
    pos, _, _ = model(batch)

    target = batch["target_attack_pos"].permute(0, 2, 1, 3)  # (B,T,n,2) -> (B,n,T,2)
    mask = batch["attack_mask"]
    loss = (pos - target).norm(dim=-1)[mask].mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_padded_players_do_not_crash_attention():
    """全選手のうち1人だけが有効、という極端なmaskでも(NaN発生などせず)動くことを確認。"""
    model = make_model()
    batch = make_synthetic_batch(n_valid_attack=1, n_valid_defend=1)
    pos, vel, acc = model(batch)

    assert torch.isfinite(pos).all()
    assert torch.isfinite(vel).all()
    assert torch.isfinite(acc).all()
