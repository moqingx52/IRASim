# MiniReal 全链路接入说明

已按计划接入 MiniReal 全链路，交付如下。

## `debug`：冒烟 vs 正式微调（必读）

`main.py` 在 **`debug: True`** 时会：

1. 把 `anno` 改成带 `-debug` 后缀（例如 `minireal_train-debug`），与正式实验目录区分；
2. `get_dataset()` 里 **train / val 都会用 `val` 划分**，即只在验证集上取样本做快速跑通；
3. `Dataset_3D` / `Dataset_MiniReal` 还会 **只保留前 10 条 sample**。

因此 **`debug: True` 只做「小样本冒烟」**，并不是在完整训练集上训练。正式微调前必须在训练 yaml（如 `configs/train/minireal/frame_ada.yaml`）里设 **`debug: False`**，否则会误以为在训练，实际只在 val 上前 10 条上迭代。

## `project_dir` 与运行目录（如 `/workspace/IRASim`）

`configs/base/data.yaml` 里 **`project_dir` 应为 `"."`**（在 IRASim 仓库根目录执行 `python main.py` 时）。若写成 `IRASim`，在仓库根下会变成 `./IRASim/pretrained_models/...`，容易找不到 SDXL VAE。服务器上若在 **`/workspace/IRASim`** 作为 cwd，同样保持 **`project_dir: "."`** 即可。

## 新增 / 修改的文件

| 路径 | 说明 |
|------|------|
| `IRASim/scripts/convert_minireal_to_irasim.py` | 将 `release/train/<ep>/` 转为 `robotdata/opensource_robotdata/minireal/{videos,annotation}/{train,val}/`；支持 `instruction.txt` / `instructions.txt`；`--rdt-actions` 可选复制 `action_rdt.npy`；`--limit` 调试用 |
| `IRASim/scripts/minireal_action_util.py` | joint→state（左臂 6 关节 + 手指归一化 gripper）、`states_to_delta_actions`、RDT 行与 test 行拼轨迹 |
| `IRASim/scripts/rollout_minireal.py` | 读 test 前 16 帧 + 第 15 行 joint + RDT `candidate_*.npy` 或 `ep.npy`；自回归 4 段（每段 15 步动作）生成 50 帧 RGB；写 `frames.npy`、`action_51x26.npy`、`meta.json` |
| `IRASim/scripts/rdt_action_txt_to_npy.py` | RDT 输出的 `<src>/<ep>/action.txt`（与 train 同格式）转为平铺 `<out>/<ep>.npy` `[51,26]`，供 `rollout_minireal.py` 读取；支持 `--only-test`、`--slice strict|last|first` |
| `IRASim/scripts/rerank_and_export.py` | 按亮度 + 帧间变化打分选候选；可选 `--add-baseline-repeat`；导出 **`--out` 根目录下** `<ep>/`（无 `submission/` 子目录）：`video.mp4`、`action.txt`、`joint.txt`、`instruction.txt` 与 `instructions.txt` |
| `IRASim/dataset/minireal.py` | `Dataset_MiniReal` 继承 `Dataset_3D`，override `_get_actions()`：MiniReal 左臂为**关节角**，动作为**关节差分**（非末端 xyz+rpy） |
| `IRASim/dataset/__init__.py` | 注册 `minireal` 分支 |
| `IRASim/models/irasim.py` | `extras==3` 时把 minireal 与 RT-1 同等对待 |
| `IRASim/configs/train/minireal/frame_ada.yaml` | `pre_encode: false`、`debug: false`（正式微调）、Bridge `evaluate_checkpoint`、`max_train_steps: 50000` 等 |
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

### 3. RDT action.txt → npy（竞赛提交前）

当 RDT 输出为与 `train/<ep>/action.txt` 同格式的目录树（如 `sample_result_rdt/<ep>/action.txt`），先转为平铺 npy 根目录，便于 `rollout_minireal.py` 读取：

```bash
python scripts/rdt_action_txt_to_npy.py \
  --src /workspace/sample_result_rdt \
  --out /workspace/sample_result_rdt_npy \
  --only-test /workspace/test
```

推荐工作区并列目录（路径可改）：`sample_result_rdt/`（原始 txt）→ `sample_result_rdt_npy/`（`{ep}.npy`）→ `irasim_rollouts/`（rollout 中间结果）→ **`data_result/`**（最终打包目录，与赛方 `sample_result` 结构一致：每 ep 下 `video.mp4`、`action.txt`、`joint.txt`、`instructions.txt`）。

### 4. Rollout

需 CUDA；checkpoint 由训练生成后改 yaml 或 `--checkpoint`。

```bash
python scripts/rollout_minireal.py \
  --config configs/evaluation/minireal/frame_ada.yaml \
  --checkpoint robotdata/opensource_robotdata/minireal/checkpoints/frame_ada/0050000.pt \
  --test-data /path/to/release/test \
  --rdt-actions /path/to/rdt_npy_root \
  --out irasim_rollouts
```

### 5. Rerank + 导出提交目录

`--out` 即为最终根目录（例如 `data_result`），其下每个 episode 一层子文件夹；**不是** `out/submission/<ep>/`。

```bash
python scripts/rerank_and_export.py \
  --rollouts irasim_rollouts \
  --test-data /path/to/release/test \
  --out data_result \
  --add-baseline-repeat
```

一键顺序也可使用 `scripts/run_minireal_submission.sh`（环境变量见脚本内注释）。

## RDT numpy 约定

- **单候选**：`rdt/<episode>.npy`，形状 `[51,26]` 或与竞赛 `action.txt` 数值列一致。
- **多候选**：`rdt/<episode>/candidate_0.npy` …，均为 `[51,26]`。
- **从 txt 生成**：`rdt_action_txt_to_npy.py` 写出平铺 `rdt/<episode>.npy`，与上条一致。

## 校验说明

- 已对新增脚本执行 `python3 -m py_compile`，语法通过。
- 当前沙箱无 numpy，未在本机跑通 convert 端到端；在装好 IRASim 依赖的环境下执行 convert / `main.py` 即可冒烟。

## Joint → 状态 / 动作约定

- **状态**：`state` 前 6 维为左臂关节 1–6（弧度）；`continuous_gripper_state` 由左手 5 指列归一化到 `[0,1]`。
- **动作**：训练用动作列为 **相邻帧关节差分** \(\Delta q\) + 当前步 gripper（与原始 `Dataset_3D` 的末端相对位姿语义不同，MiniReal 不能用 xyz+rpy 那套）。
- **原始 MiniReal 维度**：完整 joint 行为左臂 7 + 右臂 7 + 左手 6 + 右手 6 = **26 维**；当前管线只用左臂 6 + gripper，右手等在原始数据里占位较大，若以后要扩展需单独 **mask / 字段设计**，勿与左臂混用语义。
