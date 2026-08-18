"""バックボーン（計画書3.2節・3.6節）。

入力（観測窓の位置・速度） → 選手ごとのMLPエンコーダ → 時間方向エンコーダ(GRU)
→ 選手間Self-Attention → デコーダ(MLP) → ハード制約層 → 未来T_target frameの軌道。

【設計判断】計画書3.2節の図は「時間方向エンコーダで各選手の時系列を集約」した後に
「各時刻で選手間Self-Attentionを重み付け」と書かれており、集約後に時刻ごとの
attentionを行うのか字面上やや曖昧である。本実装では3.6節の「最小構成」の趣旨を
優先し、GRUで観測窓(1秒)を1本のサマリベクトルに集約してから、選手間
Self-Attentionを（時刻方向には）1回だけ適用する設計にした（観測窓が1秒と短いため、
窓内の相互作用を時刻ごとに追う必要性は薄いという判断）。窓内の時刻ごとの
相互作用まで表現したい場合は、エンコーダ直後・GRUの前に同様のSelf-Attentionを
挟む変更で対応できる。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hs_pinn.dataset import MAX_ATTACK_PLAYERS, MAX_DEFEND_PLAYERS, TARGET_FRAMES
from hs_pinn.hard_constraints import HardConstraintLayer, PitchBounds

ENTITY_FEATURES = 7  # x, y, vx, vy, is_attack, is_defend, is_ball（卒論PIGNNのノード特徴量と同じ7次元）

_KIND_ATTACK = 0
_KIND_DEFEND = 1
_KIND_BALL = 2


def _entity_features(pos: Tensor, vel: Tensor, kind: int) -> Tensor:
    """pos, vel: (B, T, n, 2) -> (B, T, n, ENTITY_FEATURES)"""
    B, T, n, _ = pos.shape
    type_onehot = torch.zeros(B, T, n, 3, device=pos.device, dtype=pos.dtype)
    type_onehot[..., kind] = 1.0
    return torch.cat([pos, vel, type_onehot], dim=-1)


class TrajectoryBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_heads: int = 4,
        target_frames: int = TARGET_FRAMES,
        a_max: float = 6.0,
        v_max: float = 9.0,
        dt: float = 1.0 / 25,
        pitch_bounds: PitchBounds | None = None,
        constraint_layer: nn.Module | None = None,
        predict_side: str = "attack",
    ) -> None:
        super().__init__()
        self.target_frames = target_frames
        if predict_side not in ("attack", "defend"):
            raise ValueError(predict_side)
        self.predict_side = predict_side

        # 選手ごとのエンコーダ（MLP、attack/defend/ballで重み共有）
        self.entity_encoder = nn.Sequential(
            nn.Linear(ENTITY_FEATURES, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 時間方向エンコーダ（GRU、観測窓を1本のサマリベクトルへ集約）
        self.temporal_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # 選手間Self-Attention（攻撃側・守備側・ボールの全エンティティ間）
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        # デコーダ（MLP、未来T_target frame分の"生の"加速度を選手ごとに出力）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_frames * 2),
        )

        # 制約層は差し替え可能（計画書6.1節の3パターン比較で
        # HardConstraintLayer / UnconstrainedIntegrationLayer を切り替える）
        self.hard_constraints = constraint_layer or HardConstraintLayer(
            a_max=a_max, v_max=v_max, dt=dt, pitch_bounds=pitch_bounds
        )

    def forward(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        attack_feat = _entity_features(
            batch["input_attack_pos"], batch["input_attack_vel"], _KIND_ATTACK
        )
        defend_feat = _entity_features(
            batch["input_defend_pos"], batch["input_defend_vel"], _KIND_DEFEND
        )
        ball_feat = _entity_features(
            batch["input_ball_pos"], batch["input_ball_vel"], _KIND_BALL
        )

        entities = torch.cat([attack_feat, defend_feat, ball_feat], dim=2)  # (B,T,n_entities,7)
        B, T, n_entities, _ = entities.shape

        embed = self.entity_encoder(entities)  # (B,T,n_entities,embed_dim)

        # GRUのために選手軸をバッチ軸へ畳み込む: (B,T,n,E) -> (B*n,T,E)
        embed_bn = embed.permute(0, 2, 1, 3).reshape(B * n_entities, T, -1)
        _, h_n = self.temporal_encoder(embed_bn)  # h_n: (1, B*n_entities, hidden_dim)
        summary = h_n.squeeze(0).view(B, n_entities, -1)  # (B, n_entities, hidden_dim)

        ball_mask = torch.ones(B, 1, dtype=torch.bool, device=summary.device)
        valid_mask = torch.cat([batch["attack_mask"], batch["defend_mask"], ball_mask], dim=1)
        key_padding_mask = ~valid_mask  # nn.MultiheadAttentionはTrue=無視の規約

        attn_out, _ = self.attention(summary, summary, summary, key_padding_mask=key_padding_mask)

        if self.predict_side == "attack":
            target_summary = attn_out[:, :MAX_ATTACK_PLAYERS]
            n_players = MAX_ATTACK_PLAYERS
            init_pos, init_vel = batch["init_position"], batch["init_velocity"]
        else:
            target_summary = attn_out[:, MAX_ATTACK_PLAYERS : MAX_ATTACK_PLAYERS + MAX_DEFEND_PLAYERS]
            n_players = MAX_DEFEND_PLAYERS
            init_pos, init_vel = batch["init_defend_position"], batch["init_defend_velocity"]

        raw = self.decoder(target_summary)  # (B, n_players, target_frames*2)
        raw_accel = raw.view(B, n_players, self.target_frames, 2)

        positions, velocities, accelerations = self.hard_constraints(raw_accel, init_pos, init_vel)
        return positions, velocities, accelerations
