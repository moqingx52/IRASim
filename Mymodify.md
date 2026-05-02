# MiniReal 全链路接入说明

已按计划接入 MiniReal 全链路，交付如下。

## 新增 / 修改的文件

| 路径 | 说明 |
|------|------|
| `IRASim/scripts/convert_minireal_to_irasim.py` | 将 `release/train/<ep>/` 转为 `robotdata/opensource_robotdata/minireal/{videos,annotation}/{train,val}/`；支持 `instruction.txt` / `instructions.txt`；`--rdt-actions` 可选复制 `action_rdt.npy`；`--limit` 调试用 |
| `IRASim/scripts/minireal_action_util.py` | joint→state（左臂 6 关节 + 手指归一化 gripper）、`states_to_delta_actions`、RDT 行与 test 行拼轨迹 |
| `IRASim/scripts/rollout_minireal.py` | 读 test 前 16 帧 + 第 15 行 joint + RDT `candidate_*.npy` 或 `ep.npy`；自回归 4 段（每段 15 步动作）生成 50 帧 RGB；写 `frames.npy`、`action_51x26.npy`、`meta.json` |
| `IRASim/scripts/rerank_and_export.py` | 按亮度 + 帧间变化打分选候选；可选 `--add-baseline-repeat`；导出 `submission/<ep>/`：`video.mp4`、`action.txt`、`joint.txt`、`instruction.txt` 与 `instructions.txt` |
| `IRASim/dataset/minireal.py` | `Dataset_MiniReal = Dataset_3D` |
| `IRASim/dataset/__init__.py` | 注册 `minireal` 分支 |
| `IRASim/models/irasim.py` | `extras==3` 时把 minireal 与 RT-1 同等对待 |
| `IRASim/configs/train/minireal/frame_ada.yaml` | `pre_encode: false`、`debug: true`、`max_train_steps: 50000` 等 |
| `IRASim/configs/evaluation/minireal/frame_ada.yaml` | 推理 / rollout 用；`evaluate_checkpoint` 占位 |

## 建议命令顺序（在 IRASim 根目录）

### 1. 数据转换

需 numpy / opencv；`bash scripts/install.sh` 后环境就绪。

```bash
python scripts/convert_minireal_to_irasim.py --src /path/to/release
```

### 2. 训练

多卡见 IRASim README。

```bash
python main.py --config configs/train/minireal/frame_ada.yaml
```

### 3. Rollout

需 CUDA；checkpoint 由训练生成后改 yaml 或 `--checkpoint`。

```bash
python scripts/rollout_minireal.py \
  --config configs/evaluation/minireal/frame_ada.yaml \
  --checkpoint robotdata/opensource_robotdata/minireal/checkpoints/frame_ada/0050000.pt \
  --test-data /path/to/release/test \
  --rdt-actions /path/to/rdt_npy_root \
  --out irasim_rollouts
```

### 4. Rerank + 导出提交目录

```bash
python scripts/rerank_and_export.py \
  --rollouts irasim_rollouts \
  --test-data /path/to/release/test \
  --out submission \
  --add-baseline-repeat
```

## RDT numpy 约定

- **单候选**：`rdt/<episode>.npy`，形状 `[51,26]` 或与竞赛 `action.txt` 数值列一致。
- **多候选**：`rdt/<episode>/candidate_0.npy` …，均为 `[51,26]`。

## 校验说明

- 已对新增脚本执行 `python3 -m py_compile`，语法通过。
- 当前沙箱无 numpy，未在本机跑通 convert 端到端；在装好 IRASim 依赖的环境下执行 convert / `main.py` 即可冒烟。

## Joint → 状态约定

`state` 的 6 维为左臂关节 1–6（弧度）；`continuous_gripper_state` 由左手 5 指列归一化到 `[0,1]`，与 `Dataset_3D` 的 `_get_actions` 一致。
