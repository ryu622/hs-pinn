import torch

from hs_pinn.soft_constraints import compactness_loss, space_control_loss
from hs_pinn.space_control import build_pitch_grid


def test_compactness_loss_is_zero_when_std_matches_target():
    # 4選手をx={0,0,10,10}に配置(標準偏差が明確な既知値になる配置)
    B, n = 1, 4
    pos = torch.zeros(B, n, 3, 2)  # (B, n_attack, T=3, 2)
    xs = torch.tensor([0.0, 0.0, 10.0, 10.0])
    pos[0, :, :, 0] = xs.unsqueeze(1)
    mask = torch.ones(B, n, dtype=torch.bool)

    target = float(xs.std(unbiased=False))
    loss = compactness_loss(pos, mask, target_std=target, frame_stride=1)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_compactness_loss_ignores_masked_players():
    B, n = 1, 4
    pos = torch.zeros(B, n, 3, 2)
    pos[0, :, :, 0] = torch.tensor([0.0, 0.0, 10.0, 999.0]).unsqueeze(1)  # 4人目は外れ値
    mask = torch.tensor([[True, True, True, False]])  # 4人目は無効

    valid_xs = torch.tensor([0.0, 0.0, 10.0])
    target = float(valid_xs.std(unbiased=False))
    loss = compactness_loss(pos, mask, target_std=target, frame_stride=1)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_compactness_loss_gradient_flows():
    pos = torch.randn(2, 6, 10, 2, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = compactness_loss(pos, mask, target_std=7.0)
    loss.backward()
    assert pos.grad is not None
    assert torch.any(pos.grad != 0)


def test_space_control_loss_favors_attack_occupying_more_area():
    grid = torch.from_numpy(build_pitch_grid(105.0, 68.0, nx=10, ny=8)).float()
    attack_mask = torch.ones(1, 3, dtype=torch.bool)
    defend_mask = torch.ones(1, 3, dtype=torch.bool)
    defend_pos = torch.full((1, 3, 5, 2), 200.0)  # ピッチ外遠方に置き、守備の影響をほぼゼロにする

    attack_pos_spread = torch.tensor(
        [[20.0, 34.0], [52.5, 34.0], [85.0, 34.0]]
    ).view(1, 3, 1, 2).expand(1, 3, 5, 2)
    attack_pos_clustered = torch.tensor(
        [[50.0, 34.0], [52.5, 34.0], [55.0, 34.0]]
    ).view(1, 3, 1, 2).expand(1, 3, 5, 2)

    loss_spread = space_control_loss(
        attack_pos_spread, defend_pos, attack_mask, defend_mask, grid, frame_stride=1
    )
    loss_clustered = space_control_loss(
        attack_pos_clustered, defend_pos, attack_mask, defend_mask, grid, frame_stride=1
    )

    # 広く散らばっている方がピッチ支配面積が大きく、L_space(=-支配度)は小さいはず
    assert loss_spread.item() < loss_clustered.item()


def test_space_control_loss_gradient_flows():
    grid = torch.from_numpy(build_pitch_grid(105.0, 68.0, nx=10, ny=8)).float()
    attack_pos = (torch.rand(2, 5, 8, 2) * 50).requires_grad_(True)
    defend_pos = torch.rand(2, 5, 8, 2) * 50
    attack_mask = torch.ones(2, 5, dtype=torch.bool)
    defend_mask = torch.ones(2, 5, dtype=torch.bool)

    loss = space_control_loss(attack_pos, defend_pos, attack_mask, defend_mask, grid, frame_stride=2)
    loss.backward()

    assert attack_pos.grad is not None
    assert torch.any(attack_pos.grad != 0)
