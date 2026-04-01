# 动作-机制耦合检查报告
## B1: 高层动作 (m, theta) 的影响
### 1. committee_size (m) 的影响

 m  action_idx  eligible_size  committee_size  unsafe_rate   latency  queue_size  timeout_rate          tps  structural_infeasible
 5          37      10.809167        5.000000          0.0 19.625060    0.000000           0.0 13934.050370                    0.0
 7         117      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
 9         197      10.955833        8.426667          0.0 24.132096    2.303333           0.0 11559.080258                    0.0

**预期**: m ↑ → committee_size ↑, structural_infeasible ↓

- committee_size 单调递增: ✅
- structural_infeasible 单调递减: ✅

### 2. theta 的影响

 theta  action_idx  eligible_size  committee_size  unsafe_rate   latency  queue_size  timeout_rate          tps  structural_infeasible
  0.45         116      11.514167        6.983333          0.0 22.513312    0.028333           0.0 12300.019512                    0.0
  0.50         117      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
  0.55         118       9.899167        6.866667          0.0 21.635671    0.773333           0.0 12742.630660                    0.0
  0.60         119       9.323333        6.466667          0.0 20.889878    8.225833           0.0 13164.982642                    0.0

**预期**: theta ↑ → unsafe_rate ↓, committee_size ↓ (更严格筛选)

- theta 与 unsafe_rate 相关性: nan (预期 <0)
- theta 与 committee_size 相关性: -0.883 (预期 <0)

## B2: 低层动作 (b, tau) 的影响
### 1. block_size (b) 的影响

  b  action_idx  eligible_size  committee_size  unsafe_rate   latency  queue_size  timeout_rate          tps  structural_infeasible
256          85      10.618333        6.931667          0.0 21.588252  263.686667           0.0 12608.410158                    0.0
320         101      10.838333        6.956667          0.0 21.937352    0.179167           0.0 12592.776870                    0.0
384         117      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
448         133      10.838333        6.956667          0.0 22.525509    0.179167           0.0 12224.902312                    0.0
512         149      10.837500        6.956667          0.0 22.826254    0.179167           0.0 12054.400462                    0.0

**预期**: b ↑ → queue_size ↓, latency ↑, tps ↑

- b 与 queue_size 相关性: -0.707 (预期 <0)
- b 与 latency 相关性: 0.999 (预期 >0)
- b 与 tps 相关性: -0.977 (预期 >0)

### 2. batch_timeout (tau) 的影响

 tau  action_idx  eligible_size  committee_size  unsafe_rate   latency  queue_size  timeout_rate          tps  structural_infeasible
  40         113      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
  60         117      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
  80         121      10.838333        6.956667          0.0 22.231430    0.179167           0.0 12404.539204                    0.0
 100         125      10.983333        6.573333          0.0 21.703221   23.930000           0.0 12662.529800                    0.0

**预期**: tau ↑ → latency ↑, timeout_rate ↓, queue_size ↓

- tau 与 latency 相关性: -0.775 (预期 >0)
- tau 与 timeout_rate 相关性: nan (预期 <0)
- tau 与 queue_size 相关性: 0.775 (预期 <0)

## 总结
### 耦合验证结果
- m → committee_size: ✅
- theta → unsafe_rate: ❌
- b → queue_size: ✅
- b → tps: ❌
- tau → latency: ❌
- tau → timeout_rate: ❌

**通过率**: 2/6 (33%)

⚠️ **部分耦合关系未通过验证，需要检查环境实现**
