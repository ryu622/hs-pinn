import torch

from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds


def make_layer(**kwargs) -> HardConstraintLayer:
    defaults = dict(a_max=6.0, v_max=9.0, dt=0.04, pitch_bounds=PitchBounds(0, 105, 0, 68))
    defaults.update(kwargs)
    return HardConstraintLayer(**defaults)


def test_acceleration_never_exceeds_a_max_even_for_extreme_raw_input():
    layer = make_layer()
    raw_accel = torch.tensor([[[1e6, -1e6]] * 50])  # (batch=1, T=50, 2), 極端な値
    init_pos = torch.zeros(1, 2)
    init_vel = torch.zeros(1, 2)

    _, _, accel = layer(raw_accel, init_pos, init_vel)

    assert torch.all(accel.abs() <= layer.a_max + 1e-4)


def test_velocity_never_exceeds_v_max():
    layer = make_layer()
    raw_accel = torch.full((1, 100, 2), 1e6)
    init_pos = torch.zeros(1, 2)
    init_vel = torch.zeros(1, 2)

    _, vel, _ = layer(raw_accel, init_pos, init_vel)

    assert torch.all(vel.abs() <= layer.v_max + 1e-4)


def test_position_never_leaves_pitch_bounds():
    layer = make_layer()
    raw_accel = torch.full((1, 200, 2), 1e6)  # 常に最大加速度でピッチ外へ突進させようとする
    init_pos = torch.tensor([[1.0, 1.0]])
    init_vel = torch.zeros(1, 2)

    pos, _, _ = layer(raw_accel, init_pos, init_vel)

    pb = layer.pitch_bounds
    assert torch.all(pos[..., 0] >= pb.x_min - 1e-4)
    assert torch.all(pos[..., 0] <= pb.x_max + 1e-4)
    assert torch.all(pos[..., 1] >= pb.y_min - 1e-4)
    assert torch.all(pos[..., 1] <= pb.y_max + 1e-4)


def test_zero_raw_accel_keeps_constant_velocity_straight_line():
    layer = make_layer()
    raw_accel = torch.zeros(1, 10, 2)
    init_pos = torch.tensor([[50.0, 30.0]])
    init_vel = torch.tensor([[2.0, 0.0]])  # 一定速度で+x方向へ移動中

    pos, vel, accel = layer(raw_accel, init_pos, init_vel)

    assert torch.allclose(accel, torch.zeros_like(accel), atol=1e-6)
    assert torch.allclose(vel, init_vel.unsqueeze(1).expand(-1, 10, -1), atol=1e-6)
    expected_x = 50.0 + 2.0 * layer.dt * torch.arange(1, 11)
    assert torch.allclose(pos[0, :, 0], expected_x, atol=1e-4)
    assert torch.allclose(pos[0, :, 1], torch.full((10,), 30.0), atol=1e-6)


def test_gradients_flow_through_the_layer():
    """ハード制約層を挟んでも生の加速度への勾配が消えないことを確認
    （tanh射影・clip・clampはいずれも区分的に微分可能）。"""
    layer = make_layer()
    raw_accel = torch.randn(2, 5, 2, requires_grad=True)
    init_pos = torch.zeros(2, 2)
    init_vel = torch.zeros(2, 2)

    pos, _, _ = layer(raw_accel, init_pos, init_vel)
    loss = pos.sum()
    loss.backward()

    assert raw_accel.grad is not None
    assert torch.any(raw_accel.grad != 0)


def test_batch_and_player_dimensions_are_preserved():
    layer = make_layer()
    batch, n_players, T = 3, 10, 20
    raw_accel = torch.randn(batch, n_players, T, 2)
    init_pos = torch.rand(batch, n_players, 2) * 50
    init_vel = torch.randn(batch, n_players, 2)

    pos, vel, accel = layer(raw_accel, init_pos, init_vel)

    assert pos.shape == (batch, n_players, T, 2)
    assert vel.shape == (batch, n_players, T, 2)
    assert accel.shape == (batch, n_players, T, 2)
