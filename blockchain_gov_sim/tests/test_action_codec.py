"""动作编解码测试。"""

from __future__ import annotations

from gov_sim.constants import ACTION_DIM
from gov_sim.env.action_codec import ActionCodec


def test_action_codec_roundtrip() -> None:
    """离散动作索引与语义动作应能双向一一映射。"""
    codec = ActionCodec()
    for idx in [0, 17, 128, 257, ACTION_DIM - 1]:
        action = codec.decode(idx)
        assert codec.encode(action) == idx
