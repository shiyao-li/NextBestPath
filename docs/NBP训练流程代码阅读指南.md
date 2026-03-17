# NBP 训练流程代码阅读指南

面向零基础读者：从 `python train_nbp.py` 到 `loss.backward()` 的完整代码路线说明。

---

## 1. 从命令行到参数对象：入口脚本与配置加载
[ \text{out} = \left\lfloor \frac{\text{in} + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}} \right\rfloor + 1 ]
### 1.1 调用链

```
train_nbp.py (__main__)
  → argparse 解析 -c/--config，默认 "nbp_default_training_config.json"
  → json_path = os.path.join(configs_dir, json_name)   # configs/nbp/xxx.json
  → params = load_params(json_path)                      # macarons.utility.macarons_utils
  → run_training_nbp(params=params)                      # next_best_path.trainers.train_nbp_model
```

- **入口文件**：[`train_nbp.py`](../train_nbp.py)（约 13–31 行）
  - `-c` / `--config`：配置文件名（不含路径），放在 `configs/nbp/` 下。
  - 未指定时使用 `nbp_default_training_config.json`。
  - 单卡时执行 `run_training_nbp(params)`；`ddp`/`jz` 为 True 时分支留空。

- **load_params**：[`macarons/utility/macarons_utils.py`](../macarons/utility/macarons_utils.py) 第 231–232 行  
  `load_params(json_name, flatten=True)` 内部：
  - 调用 `Params(json_name, flatten=True)`。
  - `Params` 定义在 [`macarons/utility/utils.py`](../macarons/utility/utils.py) 第 51–68 行：读 JSON，若 `flatten=True` 则用 `flatten_dict` 把“以下划线开头的键”（如 `_data`）下的子键展平到顶层，再 `self.__dict__.update(params)`。
  - 因此 JSON 里 `_data.data_path`、`_general_training.epochs` 等会变成 `params.data_path`、`params.epochs`。

### 1.2 NBP 配置 JSON 关键字段（configs/nbp/nbp_default_training_config.json）

| 顶层/分组 | 字段 | 含义 |
|-----------|------|------|
| _data | data_path | 训练数据根目录，如 `./data/doom_4/training_dataset` |
| _data | train_scenes, val_scenes, test_scenes | 场景名列表（NBP 里 train 常被代码用子目录名覆盖） |
| _camera_management | n_poses_in_trajectory | 每条轨迹最大步数（如 100） |
| _memory_replay | memory_dir_name | 每场景下 Memory 目录名，如 `macarons_memory` |
| _general_training | epochs | 大 epoch 数（当前 train_nbp_model 里写死 100） |
| _general_training | nbp_lr | NBP 学习率 |
| _general_training | nbp_batch_size | 从 LMDB 采样训练时的 batch 大小 |
| _general_training | db_path: | LMDB 路径（注意 JSON 里键名多了一个冒号 `db_path:`，实际路径在 train_nbp_model 里写死） |
| 顶层 | nbp_model_name, macarons_model_name | 模型命名 / MACARONS 权重文件名 |

注意：训练脚本里 LMDB 路径、epoch 数等部分写死在 `train_nbp_model.py`，改配置时需同时看代码。

---

## 2. 数据从哪里来：场景级 DataLoader 与 Memory 初始化

### 2.1 run_training_nbp 中的 DataLoader

在 [`next_best_path/trainers/train_nbp_model.py`](../next_best_path/trainers/train_nbp_model.py) 的 `run_training_nbp` 中：

- `dataset_path = params.data_path`
- 训练场景列表被**覆盖**为数据目录下所有子目录名：`scene_name = get_subfolder_names(dataset_path)`
- 调用：
  - `train_dataloader, _, _ = get_dataloader(train_scenes=scene_name, val_scenes=params.val_scenes, test_scenes=params.test_scenes, batch_size=1, ..., data_path=dataset_path)`

即：**按“场景”为单位的 DataLoader**，每个 batch 是一个场景（含 scene_name、obj_name、settings 等）。

### 2.2 get_dataloader 与 SceneDataset

- **get_dataloader**：[`macarons/utility/macarons_utils.py`](../macarons/utility/macarons_utils.py) 约 254–280 行  
  - 用 `data_path` 和 `train_scenes`/`val_scenes`/`test_scenes` 分别构造 `SceneDataset`，再包成 DataLoader。

- **SceneDataset**：[`macarons/utility/CustomDataset.py`](../macarons/utility/CustomDataset.py) 第 312–363 行  
  - `__init__(self, data_path, scene_names, ...)`：若 `scene_names` 为 None，则取 `data_path` 下所有子目录为场景名。
  - `__getitem__(idx)`：取 `scene_name = self.scene_names[idx]`，在 `data_path/scene_name/` 下找 `.obj`、`settings.json`（以及可选的 `occupied_pose.pt`），返回 `scene` 字典：`scene_name`、`obj_name`、`settings` 等。

因此：**每个样本 = 一个场景的 mesh 路径 + 设置**，不直接加载 mesh，在轨迹采集时再按需加载。

### 2.3 Memory 与 LMDB 的角色

- **setup_memory**：[`next_best_path/utility/nbp_utils.py`](../next_best_path/utility/nbp_utils.py) 第 239–249 行  
  - 对每个训练场景名，在 `scene_path = data_path/scene_name` 下建目录 `scene_path/params.memory_dir_name`（如 `macarons_memory`）。
  - 构造 `Memory(scene_memory_paths, n_trajectories=params.n_memory_trajectories, ...)`。
  - **Memory**：来自 MACARONS，用于在磁盘上按轨迹存放深度、pose、surface、occupancy 等；轨迹采集时通过 `memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)` 决定本 epoch 使用哪条轨迹的帧路径。

- **LMDB**：在 `train_nbp_model.py` 里单独打开（如 `./nbp/db/AiMDoom_insane.lmdb`），**只存 NBP 的训练经验**（当前输入、GT 障碍图、候选像素、真实 coverage gain 等），不存深度图。  
  - 轨迹采集阶段：`store_experience(db_env, experience_db)` 写入 LMDB。  
  - 经验回放阶段：`read_combined_data(db_env)` 或 `store_validation_data(db_env)` 从 LMDB 读出样本用于训练/验证。

**小结**：DataLoader 按场景迭代；每个场景在轨迹采集中会用到该场景下的 Memory 目录；NBP 的“经验”单独存在 LMDB 里，供后续 `train_experience_data` / `validation_model` 使用。

---

## 3. 轨迹采集（trajectory_collection）如何产生训练经验

### 3.1 在训练循环中的位置

在 `run_training_nbp` 里，每个大 epoch 开头：

1. `nbp.eval()`
2. `trajectory_collection(params, current_epoch, train_dataloader, db_env, pc2img_size, value_map_size, prediction_range, nbp, coverage_after_trajectory, memory, device, folder_img_path)`  
   → 遍历所有训练场景，在每场景里跑一段探索轨迹，并把经验写入 `db_env`（LMDB）。
3. 若是 epoch 0：`store_validation_data(db_env)` 抽出验证集。
4. 否则：`train_nbp(db_env, ...)` 从 LMDB 读经验训练 NBP。

即：**先采集，再训练**；采集阶段只做前向与写库，不反传。

### 3.2 trajectory_collection 的输入

- `params`：全局配置。  
- `current_epoch`：当前大 epoch。  
- `train_dataloader`：场景级 DataLoader。  
- `db_env`：LMDB 环境，用于 `store_experience`。  
- `pc2img_size`、`value_map_size`、`prediction_range`：点云/轨迹投影到 2D 的尺寸与范围。  
- `nbp`：NBP 模型（eval 模式）。  
- `memory`：每场景的 Memory 路径与轨迹编号。

### 3.3 单场景内“一步”的时序（简化）

对每个场景：

1. **加载场景**：mesh、settings、trimesh 碰撞、`setup_test_scene` 得到 gt_scene/covered_scene/surface_scene/proxy_scene，`setup_training_camera` 得到相机（含候选位姿字典 `splited_pose_space`）。
2. **初始化**：`full_pc`、`full_pc_colors` 为空，`Dijkstra_path = []`，`path_record = 0`。
3. **for pose_i in range(100)**（或到 coverage>0.95 提前结束）：
   - **观测**：`load_images_for_depth_model` 取当前视角 RGB/深度等；`obtain_depth` 得到 depth（或用 perfect depth）；用 `camera.compute_partial_point_cloud` 得到当前帧点云，拼到 `full_pc`。
   - **覆盖率**：`current_coverage = calculate_coverage_percentage(gt_scene_pc, full_pc)`，并记录。
   - **NBP 输入**：  
     - 将 `full_pc` 按 Y 分 n_pieces，投影到当前相机坐标系，再 `map_points_to_n_imgs` 得到多通道点云图；与历史轨迹投影图拼接 → `current_model_input`。  
     - `get_binary_obstacle_array` 得到 2D 障碍 GT → `current_gt_obs`。
   - **规划**：  
     - 若当前 Dijkstra 路径已走完（`path_record + 1 > len(Dijkstra_path)`）：  
       - 先把上一段路径产生的 `experiences_list` 里的经验写入 LMDB（见下）；清空 `experiences_list`。  
       - `predicted_value_map, _ = nbp(current_model_input)`；用 value map 对候选相机位置打分，Boltzmann 采样选目标位置；`generate_Dijkstra_path` 生成到该位置的路径 → `Dijkstra_path`，并往 `experiences_list` 追加 (coverage, current_model_input, current_gt_obs, pose, angle)。  
     - 若已有路径：从 `Dijkstra_path[path_record]` 取下一帧位姿，`camera.update_camera` 移动相机，再渲染多帧并更新 `full_pc`，`path_record += 1`。

4. **经验写入**：在“需要新路径”时，对上一段路径的 `experiences_list` 中每个时刻 ex，与后面时刻 ex_next 组成“未来位置→真实 coverage gain”的样本（像素坐标 + actual_coverage_gain），打包为 `experience_db`，调用 `store_experience(db_env, experience_db)` 写入 LMDB。

### 3.4 单步伪代码（核心逻辑）

```text
for pose_i in range(max_steps):
    渲染当前视角 → depth → 更新 full_pc，计算 current_coverage
    由 full_pc + 历史轨迹 → current_model_input；障碍 GT → current_gt_obs

    if 当前路径已走完:
        将上一段 experiences_list 中 (ex, ex_next) 的 actual_coverage_gain 与像素坐标写入 LMDB
        清空 experiences_list
        value_map, _ = nbp(current_model_input)
        根据 value_map 选目标位姿，Dijkstra 生成新路径，并往 experiences_list 追加当前状态

    从 Dijkstra_path 取下一帧，移动相机，更新 full_pc；path_record += 1
```

---

## 4. 从 LMDB 读经验到 NBP 前向、loss、反向传播

### 4.1 一轮“经验回放”的入口：train_nbp

在 [`next_best_path/utility/nbp_utils.py`](../next_best_path/utility/nbp_utils.py) 第 430–468 行：

- `training_set_db = read_combined_data(db_env)`（或首轮 `sample_m=None` 全量读）。
- `validaion_data` 在首个 epoch 已通过 `store_validation_data(db_env)` 得到，这里直接使用。
- 内层循环 `for epoch in range(num_epochs)`（如 5 个小 epoch）：
  - `train_loss = train_experience_data(training_set_db, params, optimizer, nbp, device, current_epoch)`
  - `validation_loss = validation_model(validation_data, params, nbp, device)`
  - `lr_scheduler.step(validation_loss)`

即：**从 LMDB 读出一大份 training_set_db + 固定 validation_data，然后多轮小 epoch 用同一份数据训练/验证。**

### 4.2 read_combined_data

同文件第 101–141 行：  
在 LMDB 里遍历所有 key，若 `sample_m is None` 则全部返回；否则从“前面一段”随机抽一批 + “最后 sample_m 条”合并，返回 list of experience dict。每条 experience 包含：`current_model_input`、`current_gt_2d_layout`、`target_value_map_pixel`、`actual_coverage_gain`、`pose_i` 等。

### 4.3 train_experience_data：前向 + loss + 反向

第 340–394 行：

- 按 `params.nbp_batch_size` 从 `training_set_db` 里取 batch；每条样本过滤 `pose_i`（如 pose_i>10 且 current_epoch==1 或 current_epoch>1）。
- 从每条 data 取出：
  - `current_model_input` → 输入图像
  - `current_gt_2d_layout` → 障碍 GT
  - `target_value_map_pixel` → 像素坐标 (batch_idx, y, x, channel) 等
  - `actual_coverage_gain` → 真实 coverage gain
- `predicted_value_map, predicted_obs_map = nbp(input_images)`  
  - `out1` = value map（多通道），`out2` = 障碍图（Sigmoid）。
- 用 `batch_indices`、`batch_coords` 在 `predicted_value_map` 上索引出预测值 `pred_values`。
- **loss**：`batch_loss = nbp.loss(pred_values, batch_gt_pixels, predicted_obs_map, gt_obs)`  
  - NBP 的 `loss` 在 [`next_best_path/networks/nbp_model.py`](../next_best_path/networks/nbp_model.py) 第 162–172 行：多任务不确定性加权（MSE for value + BCE for obstacle + log_var 项）。
- `scaler.scale(batch_loss).backward()`，每 `accumulation_steps` 次做 `scaler.step(optimizer)`、`optimizer.zero_grad()`。

### 4.4 validation_model

第 291–337 行：与 train_experience_data 类似地组 batch、做 `nbp(input_images)`、用坐标索引 pred_values，但用固定 MSE+BCE 算 loss（不用 nbp.loss），且**无 backward**，只累加求平均 validation loss。

### 4.5 NBP 网络结构（nbp_model.py）与 loss

- **输入**：5 通道（点云投影 4 块 + 历史轨迹 1 通道），形状如 (B, 5, H, W)。
- **编码器**：Conv1(5→64) → MaxPool → Conv2(64→128) → … → Conv5(512→1024)，逐层下采样。
- **解码器 1（value map）**：Up5_1 + Attention + skip → Up4_1 + Attention → Final1 → **out1**，通道数 8。
- **解码器 2（obstacle map）**：Up5_2 → … → Up2_2 + Attention → Final2(Conv + Sigmoid) → **out2**，1 通道。
- **loss**：可学习 `log_vars` 两个参数，用 MSE(pred_values, target_gain) 与 BCE(pred_obs, gt_obs) 加 1/σ² 权重和 log_var 正则项（多任务不确定性加权）。

训练时实际调用的是 **nbp_utils 里组 batch + 索引 + nbp.loss**，而不是 nbp_model 里单独算 MSE/BCE；验证时用 nbp_utils 里的 MSE+BCE 实现。

---

## 5. 训练循环整体节奏与模型保存

在 [`next_best_path/trainers/train_nbp_model.py`](../next_best_path/trainers/train_nbp_model.py) 中：

- 外层：`for current_epoch in range(0, 100)`（大 epoch 数写死）。
- 每个大 epoch：
  1. **轨迹采集**：`nbp.eval()` → `trajectory_collection(...)`，刷新 LMDB 中的经验。
  2. **Epoch 0**：只做 `validaion_data = store_validation_data(db_env)`，不训练。
  3. **Epoch ≥ 1**：`nbp.train()` → `training_loss, average_validation_loss = train_nbp(db_env, params, optimizer, nbp, device, ..., validaion_data)`。  
     - `train_nbp` 内：`read_combined_data` → 5 个小 epoch，每轮 `train_experience_data` + `validation_model` + lr_scheduler.step。
  4. **保存**：  
     - 若 `average_validation_loss < best_validation_loss`：保存为 `AiMDoom_insane_best_val.pth`（路径在 `nbp_weights_path` 下）。  
     - 每 3 个 epoch 再存一次带 epoch 号的 checkpoint，如 `AiMDoom_insane_xx_model_Epoch3.pth`。  
  5. 将本 epoch 的 training_loss、validation_loss 写入 JSON（`training_process_path`）。

测试脚本通过 config 里的 `model_name` 或约定好的文件名（如 `*_best_val.pth`）加载权重。

---

## 6. 阅读顺序 Cheat Sheet（按文件）

| 顺序 | 文件 | 看哪里 | 脑子里想的三个问题 | 看完能回答 |
|------|------|--------|---------------------|------------|
| 1 | `train_nbp.py` | 全文 | 入口如何解析配置？config 从哪来？谁真正开训？ | 命令行到 `run_training_nbp(params)` 的路径 |
| 2 | `macarons/utility/utils.py` | Params、flatten_dict | JSON 如何变成 params.xxx？哪些键会展平？ | params.data_path 对应 JSON 里哪一项 |
| 3 | `configs/nbp/nbp_default_training_config.json` | _data、_general_training、顶层 | 数据路径、batch、学习率、epoch 在哪？ | 改数据目录/学习率应改哪些键 |
| 4 | `next_best_path/trainers/train_nbp_model.py` | run_training_nbp 整体 | DataLoader 怎么来的？LMDB 何时打开？大 epoch 里先做什么后做什么？ | 训练主循环的先后顺序 |
| 5 | `macarons/utility/CustomDataset.py` | SceneDataset.__init__, __getitem__ | 一个“样本”是什么？从哪里读 mesh/settings？ | 为什么说是“场景级”数据 |
| 6 | `next_best_path/utility/nbp_utils.py` | setup_memory | Memory 目录建在哪？和 LMDB 有何区别？ | Memory 存什么、LMDB 存什么 |
| 7 | `next_best_path/utility/nbp_utils.py` | trajectory_collection 从头到尾 | 每一步如何更新点云？何时调 NBP？何时写 LMDB？ | 经验样本是怎么构造的 |
| 8 | `next_best_path/utility/nbp_utils.py` | read_combined_data, store_experience | 经验以什么结构存？读出来是什么？ | LMDB 里一条记录有哪些键 |
| 9 | `next_best_path/utility/nbp_utils.py` | train_experience_data, validation_model | batch 里有哪些张量？loss 在哪算？谁调 backward？ | 从 batch 到 loss.backward() 的步骤 |
| 10 | `next_best_path/networks/nbp_model.py` | NBP.forward, NBP.loss | 输入输出维度？两个头各是什么？loss 公式？ | value map 和 obstacle map 的通道与含义 |

---

文档版本：与当前代码库一致；若脚本中硬编码路径或 epoch 数有改动，以代码为准。
