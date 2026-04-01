"""固定高层模板选择器，用于前沿评估。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.hierarchical.controller import count_legal_high_templates, resolve_template
from gov_sim.hierarchical.spec import HighLevelAction, HierarchicalActionCodec


class FixedTemplateSelector:
    """评估时固定高层模板，用于前沿分析。
    
    与 EpisodeSampledTemplateSelector 不同，该选择器在整个评估过程中
    始终返回指定的固定模板，不进行随机采样。
    
    关键：记录 requested vs applied 模板，确保 remap 可追溯。
    """

    def __init__(
        self,
        template: HighLevelAction,
        codec: HierarchicalActionCodec | None = None,
    ) -> None:
        """初始化固定模板选择器。
        
        Args:
            template: 固定的高层模板 (m, theta)
            codec: 动作编解码器
        """
        self.codec = codec or HierarchicalActionCodec()
        self.fixed_template = template
        self.current_template = template
        self.last_requested_template: HighLevelAction | None = None
        self.last_resolved_template: HighLevelAction | None = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = self.codec.high_action_repr(template)
        
        # Remap 追踪
        self.was_remapped = False
        self.remap_reason = ""
        self.requested_template_idx = self.codec.encode_high(template)
        self.applied_template_idx = self.requested_template_idx

    def reset(self, seed: int | None = None) -> None:
        """重置选择器状态。
        
        Args:
            seed: 忽略，保持接口兼容性
        """
        del seed  # 固定模板不需要随机性
        self.current_template = self.fixed_template
        self.last_requested_template = self.fixed_template
        self.last_resolved_template = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = self.codec.high_action_repr(self.fixed_template)
        
        # 重置 remap 追踪
        self.was_remapped = False
        self.remap_reason = ""
        self.requested_template_idx = self.codec.encode_high(self.fixed_template)
        self.applied_template_idx = self.requested_template_idx

    def template_for_step(self, env: BlockchainGovEnv) -> HighLevelAction:
        """返回当前步的固定模板。
        
        Args:
            env: 环境实例
            
        Returns:
            固定的高层模板
        """
        self.last_requested_template = self.fixed_template
        self.last_num_legal_high_templates = count_legal_high_templates(env=env, codec=self.codec)
        
        # 尝试使用固定模板，如果不合法则 remap 到最近的合法模板
        from gov_sim.hierarchical.controller import build_high_level_mask, _resolve_high_action
        
        mask = build_high_level_mask(env=env, codec=self.codec)
        requested_idx = self.codec.encode_high(self.fixed_template)
        
        _, template, _, source, _, policy_high_template = _resolve_high_action(
            env=env,
            codec=self.codec,
            requested_idx=requested_idx,
        )
        
        # 检查是否发生 remap
        if template != self.fixed_template:
            self.was_remapped = True
            self.remap_reason = "legality_constraint"
            self.applied_template_idx = self.codec.encode_high(template)
        else:
            self.was_remapped = False
            self.remap_reason = ""
            self.applied_template_idx = self.requested_template_idx
        
        self.current_template = template
        self.last_resolved_template = template
        self.last_policy_high_template = policy_high_template
        return self.current_template

    def on_step_complete(self) -> None:
        """步骤完成回调，保持接口兼容性。"""
        return None

    def select(self, env: BlockchainGovEnv) -> tuple[int, HighLevelAction, np.ndarray]:
        """选择固定模板（兼容 HighLevelInferenceAdapter 接口）。
        
        Args:
            env: 环境实例
            
        Returns:
            (action_idx, high_action, mask)
        """
        from gov_sim.hierarchical.controller import build_high_level_mask
        
        template = self.template_for_step(env)
        action_idx = self.codec.encode_high(template)
        mask = build_high_level_mask(env=env, codec=self.codec)
        return action_idx, template, mask
    
    def get_remap_info(self) -> dict[str, Any]:
        """获取 remap 信息。
        
        Returns:
            包含 requested/applied 模板信息的字典
        """
        requested = self.fixed_template
        applied = self.current_template
        
        return {
            "requested_template_idx": self.requested_template_idx,
            "requested_m": requested.m,
            "requested_theta": requested.theta,
            "applied_template_idx": self.applied_template_idx,
            "applied_m": applied.m,
            "applied_theta": applied.theta,
            "was_remapped": self.was_remapped,
            "remap_reason": self.remap_reason,
        }
