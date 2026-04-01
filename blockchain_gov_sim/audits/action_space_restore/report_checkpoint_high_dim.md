# Checkpoint 高层动作空间维度审计报告

**审计时间**: 1773849765.6172845

## 审计目标

核验 `outputs/formal_multiseed/hierarchical/formal_final_seed*` 下的训练结果，
确认是否基于 12 维高层动作空间（包含 m=5 的 nominal 模板）训练。

## 审计方法

1. 扫描每个 seed 的 `config_snapshot.json`
2. 检查 `stage2_high_train/train_audit.json` 中的 action_frequency 和 executed_high_template_distribution
3. 检查 `stage3_high_refine/round_*_high_train_audit.json` 中的相同字段
4. 判断是否出现 m=5 的模板（nominal 模板的标志）

## 审计结果汇总

| Seed | Stage2 存在 | Stage3 存在 | Stage2 有 m=5 | Stage3 有 m=5 | m=5 比例 (Stage2) | m=5 比例 (Stage3) | 12 维确认 |
|------|------------|------------|---------------|---------------|-------------------|-------------------|----------|
| formal_final_seed42 | ✓ | ✓ | ✓ | ✓ | 0.0502 | 0.0508 | ✓ |
| formal_final_seed43 | ✓ | ✓ | ✓ | ✓ | 0.0440 | 0.0449 | ✓ |
| formal_final_seed44 | ✓ | ✓ | ✓ | ✓ | 0.0510 | 0.0488 | ✓ |
| formal_final_seed45 | ✓ | ✓ | ✓ | ✓ | 0.0461 | 0.0527 | ✓ |
| formal_final_seed46 | ✓ | ✓ | ✓ | ✓ | 0.0510 | 0.0527 | ✓ |

## 详细分析

### formal_final_seed42

**结论**: ✅ 确认为 12 维高层动作空间训练结果

**证据来源**:
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed42/config_snapshot.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed42/stage2_high_train/train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/round_01_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/round_02_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/round_03_high_train_audit.json`

**详细信息**:
- Config 存在: 是
- Stage2 训练审计存在: 是
- Stage3 训练审计存在: 是
- Stage2 中 m=5 出现比例: 0.0502
- Stage3 中 m=5 出现比例: 0.0508

### formal_final_seed43

**结论**: ✅ 确认为 12 维高层动作空间训练结果

**证据来源**:
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed43/config_snapshot.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed43/stage2_high_train/train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/round_01_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/round_02_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/round_03_high_train_audit.json`

**详细信息**:
- Config 存在: 是
- Stage2 训练审计存在: 是
- Stage3 训练审计存在: 是
- Stage2 中 m=5 出现比例: 0.0440
- Stage3 中 m=5 出现比例: 0.0449

### formal_final_seed44

**结论**: ✅ 确认为 12 维高层动作空间训练结果

**证据来源**:
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed44/config_snapshot.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed44/stage2_high_train/train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/round_01_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/round_02_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/round_03_high_train_audit.json`

**详细信息**:
- Config 存在: 是
- Stage2 训练审计存在: 是
- Stage3 训练审计存在: 是
- Stage2 中 m=5 出现比例: 0.0510
- Stage3 中 m=5 出现比例: 0.0488

### formal_final_seed45

**结论**: ✅ 确认为 12 维高层动作空间训练结果

**证据来源**:
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed45/config_snapshot.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed45/stage2_high_train/train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/round_01_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/round_02_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/round_03_high_train_audit.json`

**详细信息**:
- Config 存在: 是
- Stage2 训练审计存在: 是
- Stage3 训练审计存在: 是
- Stage2 中 m=5 出现比例: 0.0461
- Stage3 中 m=5 出现比例: 0.0527

### formal_final_seed46

**结论**: ✅ 确认为 12 维高层动作空间训练结果

**证据来源**:
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed46/config_snapshot.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed46/stage2_high_train/train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/round_01_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/round_02_high_train_audit.json`
- `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/round_03_high_train_audit.json`

**详细信息**:
- Config 存在: 是
- Stage2 训练审计存在: 是
- Stage3 训练审计存在: 是
- Stage2 中 m=5 出现比例: 0.0510
- Stage3 中 m=5 出现比例: 0.0527

## 最终结论

- 总共扫描: 5 个 seed
- 确认为 12 维: 5 个
- 未确认为 12 维: 0 个

✅ **所有 formal_final_seed* 结果均确认为 12 维高层动作空间训练结果，可以继续使用。**
