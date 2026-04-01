"""测试高层动作空间恢复为完整 12 模板。

验证：
1. HIGH_LEVEL_TEMPLATES 包含 12 个模板 (m∈{5,7,9}, theta∈{0.45,0.50,0.55,0.60})
2. 高层动作空间维度为 12
3. 不再有 backstop 模板
4. 动作编解码正确
5. 高层 mask 维度正确
"""

from __future__ import annotations

import numpy as np

from gov_sim.hierarchical.spec import (
    HIGH_LEVEL_M_VALUES,
    HIGH_LEVEL_TEMPLATES,
    HIGH_LEVEL_THETA_VALUES,
    HierarchicalActionCodec,
)


def test_high_level_m_values():
    """验证 m 值恢复为完整集合。"""
    assert HIGH_LEVEL_M_VALUES == (5, 7, 9), f"Expected (5, 7, 9), got {HIGH_LEVEL_M_VALUES}"
    print("✓ HIGH_LEVEL_M_VALUES 正确: (5, 7, 9)")


def test_high_level_theta_values():
    """验证 theta 值集合。"""
    assert HIGH_LEVEL_THETA_VALUES == (0.45, 0.50, 0.55, 0.60), f"Expected (0.45, 0.50, 0.55, 0.60), got {HIGH_LEVEL_THETA_VALUES}"
    print("✓ HIGH_LEVEL_THETA_VALUES 正确: (0.45, 0.50, 0.55, 0.60)")


def test_high_level_templates_count():
    """验证高层模板数量为 12。"""
    assert len(HIGH_LEVEL_TEMPLATES) == 12, f"Expected 12 templates, got {len(HIGH_LEVEL_TEMPLATES)}"
    print(f"✓ HIGH_LEVEL_TEMPLATES 数量正确: {len(HIGH_LEVEL_TEMPLATES)}")


def test_high_level_templates_content():
    """验证高层模板内容完整。"""
    expected_templates = [
        (5, 0.45), (5, 0.50), (5, 0.55), (5, 0.60),
        (7, 0.45), (7, 0.50), (7, 0.55), (7, 0.60),
        (9, 0.45), (9, 0.50), (9, 0.55), (9, 0.60),
    ]
    assert HIGH_LEVEL_TEMPLATES == tuple(expected_templates), f"Templates mismatch"
    print("✓ HIGH_LEVEL_TEMPLATES 内容正确")
    for m, theta in HIGH_LEVEL_TEMPLATES:
        print(f"  - ({m}, {theta:.2f})")


def test_codec_high_dim():
    """验证 codec 高层维度为 12。"""
    codec = HierarchicalActionCodec()
    assert codec.high_dim == 12, f"Expected high_dim=12, got {codec.high_dim}"
    print(f"✓ Codec high_dim 正确: {codec.high_dim}")


def test_codec_no_backstop():
    """验证 codec 不再有 backstop 相关属性。"""
    codec = HierarchicalActionCodec()
    assert not hasattr(codec, "backstop_high_action"), "Codec should not have backstop_high_action"
    assert not hasattr(codec, "executable_high_actions"), "Codec should not have executable_high_actions"
    print("✓ Codec 已移除 backstop 相关属性")


def test_codec_encode_decode():
    """验证动作编解码正确。"""
    codec = HierarchicalActionCodec()
    
    # 测试所有 12 个模板
    for idx in range(12):
        action = codec.decode_high(idx)
        encoded_idx = codec.encode_high(action)
        assert encoded_idx == idx, f"Encode/decode mismatch at idx={idx}"
    
    print("✓ 所有 12 个模板编解码正确")


def test_high_level_mask_dimension():
    """验证高层 mask 维度为 12。"""
    from gov_sim.hierarchical.controller import build_high_level_mask
    from gov_sim.env.gov_env import BlockchainGovEnv
    import yaml
    
    # 加载默认配置
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    env = BlockchainGovEnv(config)
    env.reset(seed=42)
    codec = HierarchicalActionCodec()
    
    mask = build_high_level_mask(env=env, codec=codec)
    assert mask.shape == (12,), f"Expected mask shape (12,), got {mask.shape}"
    print(f"✓ 高层 mask 维度正确: {mask.shape}")


def test_all_templates_accessible():
    """验证所有 12 个模板都可访问。"""
    codec = HierarchicalActionCodec()
    
    # 验证所有模板都在 high_actions 中
    assert len(codec.high_actions) == 12, f"Expected 12 high_actions, got {len(codec.high_actions)}"
    
    # 验证包含所有 m 和 theta 组合
    m_values = set()
    theta_values = set()
    for action in codec.high_actions:
        m_values.add(action.m)
        theta_values.add(action.theta)
    
    assert m_values == {5, 7, 9}, f"Expected m values {{5, 7, 9}}, got {m_values}"
    assert theta_values == {0.45, 0.50, 0.55, 0.60}, f"Expected theta values {{0.45, 0.50, 0.55, 0.60}}, got {theta_values}"
    
    print("✓ 所有 12 个模板都可访问")
    print(f"  - m 值: {sorted(m_values)}")
    print(f"  - theta 值: {sorted(theta_values)}")


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("测试高层动作空间恢复")
    print("=" * 60)
    
    tests = [
        test_high_level_m_values,
        test_high_level_theta_values,
        test_high_level_templates_count,
        test_high_level_templates_content,
        test_codec_high_dim,
        test_codec_no_backstop,
        test_codec_encode_decode,
        test_high_level_mask_dimension,
        test_all_templates_accessible,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n运行: {test.__name__}")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 测试错误: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
