import os
import gc

import random
import torch
import ast

import torch.nn as nn
import torch.optim as optim
import matplotlib.patches as patches

from collections import deque
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.jit import script
from itertools import islice

import trimesh
from navi.utility.navi_utils import *
from navi.utility.utils import *
from macarons.utility.macarons_utils import *
from macarons.trainers.train_macarons import setup_scene, setup_camera
from macarons.testers.scene import setup_test_scene, setup_test_2d_camera
from macarons.utility.long_term_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms

import lmdb
import msgpack
import msgpack_numpy as m

def store_experience(env, data):
    with env.begin(write=True) as txn:
        key = f"{int(time.time()*1000):012d}".encode()  # 生成基于毫秒的时间戳键
        # 序列化数据
        packed_data = msgpack.packb({
            'current_model_input': data['current_model_input'].cpu().numpy(),
            'current_gt_2d_layout': data['current_gt_2d_layout'].cpu().numpy(),
            'target_heatmap_pixel': data['target_heatmap_pixel'].cpu().numpy(),
            'actual_path_gain': data['actual_path_gain'].cpu().numpy(),
            'pose_i': np.array(data['pose_i'])
        }, use_bin_type=True)
        txn.put(key, packed_data)

def store_validation_data_readonly(env, num=600*2):
    selected_data = []
    with env.begin(write=False) as txn:  # 更改为不需要写权限，因为不再删除数据
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)  # 计算总条目数
        print("Number of total data in the database:", total_entries)
        
        # 根据需要抽取的数据数量 k 计算间隔 n
        n = math.ceil(total_entries / num)  # 计算间隔
        
        # 遍历数据，按计算出的间隔抽取数据
        count = 0
        for i, (key, value) in enumerate(cursor):
            if count % n == 0 and len(selected_data) < num:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))  # 确保使用合适的解码函数
                if len(selected_data) == num:  # 如果已经抽取足够数量的数据，停止遍历
                    break
            count += 1

    return selected_data

def read_random_data_readonly(env, num_samples=64):
    data = []
    indices = set(random.sample(range(env.stat()['entries']), num_samples))  # 预先选择索引
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i in indices:
                data.append(msgpack.unpackb(value, object_hook=m.decode))
                if len(data) == num_samples:
                    break

    return data

def store_validation_data(env, num=600*2):
    selected_data = []
    delete_keys = []
    with env.begin(write=True) as txn:  # 需要写权限以删除数据
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)  # 计算总条目数
        print("Number of total data in the database:", total_entries)
        
        # 根据需要抽取的数据数量 k 计算间隔 n
        n = math.ceil(total_entries / num)  # 计算间隔
        
        # 遍历数据，按计算出的间隔抽取数据
        count = 0
        for i, (key, value) in enumerate(cursor):
            if count % n == 0 and len(selected_data) < num:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))
                delete_keys.append(key)
                if len(selected_data) == num:  # 如果已经抽取足够数量的数据，停止遍历
                    break
            count += 1

        # 删除选定的数据
        for key in delete_keys:
            txn.delete(key)

    return selected_data


def read_combined_data(env, sample_m=2304*2):
    selected_data = []
    last_data = []
    total_entries = 0
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        # 计算数据库中的总条目数
        for key, value in cursor:
            total_entries += 1
        print("number of total data in the database:", total_entries)
        
        # 如果sample_m未提供，返回数据库中的所有数据
        if sample_m is None:
            cursor.first()
            return [msgpack.unpackb(value, object_hook=m.decode) for key, value in cursor]
        
        n = total_entries - sample_m  # 设置n为总条目数减去m

        # 如果n小于0，意味着m值可能过大
        if n < 0:
            n = 1
        
        # 读取前n条数据的随机十分之一
        cursor.first()
        sample_size = 2176*2
        sample_indices = random.sample(range(n), min(sample_size, n))  # 确保样本大小不会超过 n
        count = 0
        for i, (key, value) in enumerate(cursor):
            if i in sample_indices:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))
            count += 1
            if count >= n:
                break
        
        # 读取最后m条数据
        if cursor.last():
            last_data.append(msgpack.unpackb(cursor.value(), object_hook=m.decode))
            for _ in range(sample_m - 1):
                if cursor.prev():
                    last_data.append(msgpack.unpackb(cursor.value(), object_hook=m.decode))
            last_data.reverse()  # 翻转列表以返回正确顺序的数据

    # 合并两组数据
    combined_data = selected_data + last_data
    return combined_data

class Experience:
    def __init__(self, current_model_input, current_gt_2d_layout, target_heatmap_pixel, actual_path_gain):
        self.current_model_input = current_model_input
        self.current_gt_2d_layout = current_gt_2d_layout
        self.target_heatmap_pixel = target_heatmap_pixel
        self.actual_path_gain = actual_path_gain

    def __iter__(self):
        return iter((self.current_model_input,
                     self.current_gt_2d_layout, self.target_heatmap_pixel, self.actual_path_gain))

class ReplayBuffer:
    def __init__(self, capacity=60000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer.clear()
    
class LogExplorationLoss(nn.Module):
    def __init__(self, scaling_factor=100):
        super(LogExplorationLoss, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, pred_gain, actual_gain):
        # 将增益值缩放
        pred_gain_scaled = self.scaling_factor * pred_gain
        actual_gain_scaled = self.scaling_factor * actual_gain

        # 计算均方误差
        mse = (pred_gain_scaled - actual_gain_scaled) ** 2

        return mse.mean()
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.2):
        """
        patience: 
        min_delta: 
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if isinstance(m, nn.Linear):  # or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
        # torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        # torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.weight, 0., 0.005) #0.005)
        # m.bias.data.fill(0.01)

def initialize_navi(params, navi, torch_seed, initialize, 
                     pretrained, ddp_rank):
    model_name = params.navi_model_name
    start_epoch = 0
    best_loss = 10000.
    
    # Weight initialization if needed
    if initialize:

        navi.apply(init_weights)
                
    else: 
        pass
    
    # optimizer = optim.Adam(navi.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    
    optimizer = optim.AdamW(navi.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    # if params.opt_name == "adam":
    #     optimizer = optim.Adam(navi.parameters(), lr=params.navi.lr)
    
    # elif params.opt_name == "adagrad":
    #     optimizer = optim.Adagrad(navi.parameters(), lr=params.navi_lr)
    
    # elif params.opt_name == "rmsprop":
    #     optimizer = optim.RMSprop(navi.parameters(), lr=params.navi_lr, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0)

    return navi, optimizer, best_loss, start_epoch

def setup_macarons(params, macarons_model_path, device):
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_macarons_path, device=device, learn_pose=params.learn_pose)
    trained_weights = torch.load(macarons_model_path, map_location=device)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True) 
    return macarons

def setup_memory(params, train_set, train_dataloader):
    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in train_set:
        scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0)
    return memory


# def setup_memory_and_macarons(params, macarons_model_path, train_dataloader, device):
#     macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_macarons_path, device=device, learn_pose=params.learn_pose)
#     trained_weights = torch.load(macarons_model_path, map_location=device)
#     macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True) 
#     # Creating memory
#     print("\nUsing memory folders", params.memory_dir_name)
#     scene_memory_paths = []
#     for scene_name in params.train_scenes:
#         scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
#         scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
#         scene_memory_paths.append(scene_memory_path)
#     memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
#                     current_epoch=0)
#     return macarons, memory

def augment_data(gt_layout, input_tensor):
    # 定义一个完整的变换过程
    transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)
        ], p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ])
    ])
    
    # 使用RandomApply对整个变换过程添加一定概率保持不变
    full_transform = transforms.RandomApply([transform], p=0.4)

    # 确保使用相同的变换来保持数据一致性
    seed = torch.random.initial_seed()
    torch.manual_seed(seed)
    augmented_gt_layout = full_transform(gt_layout)

    torch.manual_seed(seed)
    augmented_input = full_transform(input_tensor)

    return augmented_gt_layout, augmented_input


# def train_experience_data(heatmap_exp, layout_exp, params, optimizer, navi, device):
#     training_loss = []
#     total_heatmap_batches = len(heatmap_exp) // params.heatmap_batch_size + (1 if len(heatmap_exp) % params.heatmap_batch_size > 0 else 0)
    
#     criterion_maps = nn.MSELoss()  # MSE用于热图预测
#     criterion_layout = nn.BCELoss()  # BCE用于布局预测
#     scaler = GradScaler()

#     # 计算每个heatmap batch对应的layout样本数
#     layout_samples_per_batch = len(layout_exp) // total_heatmap_batches
#     layout_remainder = len(layout_exp) % total_heatmap_batches

#     for batch_index in range(total_heatmap_batches):
#         optimizer.zero_grad()

#         with autocast():
#             # 处理heatmap数据
#             start_index = batch_index * params.heatmap_batch_size
#             end_index = min(start_index + params.heatmap_batch_size, len(heatmap_exp))
#             heatmap_batch = heatmap_exp[start_index:end_index]
            
#             batch_start_grids, batch_previous_trajectories, _, batch_target_locations, batch_angles, batch_gains = zip(*heatmap_batch)
            
#             batch_start_grids = torch.cat(batch_start_grids).to(device)
#             batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
#             batch_target_locations = torch.cat(batch_target_locations).to(device)
#             batch_angles = torch.cat(batch_angles).to(device)
#             batch_gains = torch.tensor(batch_gains, dtype=torch.float32, device=device)

#             output_maps, _ = navi(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1).to(device))
            
#             indices = torch.arange(len(batch_angles), device=device)
#             predicted_potentials = output_maps[indices, batch_angles, batch_target_locations[:, 0], batch_target_locations[:, 1]]
#             loss_maps = criterion_maps(predicted_potentials, batch_gains)

#             # 处理layout数据
#             layout_start = batch_index * layout_samples_per_batch
#             layout_end = layout_start + layout_samples_per_batch + (1 if batch_index < layout_remainder else 0)
#             layout_batch = layout_exp[layout_start:layout_end]
            
#             if layout_batch:
#                 batch_start_grids, batch_previous_trajectories, batch_gt_2d_layouts, _, _, _ = zip(*layout_batch)
                
#                 batch_start_grids = torch.cat(batch_start_grids).to(device)
#                 batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
#                 batch_gt_2d_layouts = torch.cat(batch_gt_2d_layouts).to(device)

#                 if batch_gt_2d_layouts.dtype != torch.float32:
#                     batch_gt_2d_layouts = batch_gt_2d_layouts.float()

#                 # 应用数据增强
#                 # augmented_gt_layouts, augmented_inputs = augment_data(batch_gt_2d_layouts, torch.cat((batch_start_grids, batch_previous_trajectories), dim=1))

#                 _, output_layout = navi(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1))
#                 loss_layout = criterion_layout(output_layout, batch_gt_2d_layouts.to(device))
#             else:
#                 loss_layout = torch.tensor(0.0, device=device)

#             # 计算总损失
#             batch_loss = loss_maps + loss_layout

#         scaler.scale(batch_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
# 27号版本
# def train_experience_data(heatmap_exp, layout_exp, params, optimizer, navi, device):
#     training_loss = []
#     total_heatmap_batches = len(heatmap_exp) // params.heatmap_batch_size + (1 if len(heatmap_exp) % params.heatmap_batch_size > 0 else 0)
    
#     criterion_maps = nn.MSELoss()  # MSE用于热图预测
#     criterion_layout = nn.BCELoss()  # BCE用于布局预测
#     scaler = GradScaler()

#     # 计算每个heatmap batch对应的layout样本数
#     layout_samples_per_batch = len(layout_exp) // total_heatmap_batches
#     layout_remainder = len(layout_exp) % total_heatmap_batches

#     for batch_index in range(total_heatmap_batches):
#         optimizer.zero_grad()

#         # 处理heatmap数据
#         start_index = batch_index * params.heatmap_batch_size
#         end_index = min(start_index + params.heatmap_batch_size, len(heatmap_exp))
#         heatmap_batch = heatmap_exp[start_index:end_index]
        
#         batch_start_grids, batch_previous_trajectories, _, batch_target_locations, batch_angles, batch_gains = zip(*heatmap_batch)
        
#         batch_start_grids = torch.cat(batch_start_grids).to(device)
#         batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
#         batch_target_locations = torch.cat(batch_target_locations).to(device)
#         batch_angles = torch.cat(batch_angles).to(device)
#         batch_gains = torch.tensor(batch_gains, dtype=torch.float32, device=device)

#         # 开启 autocast
#         with autocast():
#             output_maps, _ = navi(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1).to(device))
            
#             indices = torch.arange(len(batch_angles), device=device)
#             predicted_potentials = output_maps[indices, batch_angles, batch_target_locations[:, 0], batch_target_locations[:, 1]]
#             loss_maps = criterion_maps(predicted_potentials, batch_gains)

#         # 处理layout数据
#         layout_start = batch_index * layout_samples_per_batch
#         layout_end = layout_start + layout_samples_per_batch + (1 if batch_index < layout_remainder else 0)
#         layout_batch = layout_exp[layout_start:layout_end]
        
#         if layout_batch:
#             batch_start_grids, batch_previous_trajectories, batch_gt_2d_layouts, _, _, _ = zip(*layout_batch)
            
#             batch_start_grids = torch.cat(batch_start_grids).to(device)
#             batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
#             batch_gt_2d_layouts = torch.cat(batch_gt_2d_layouts).to(device)

#             if batch_gt_2d_layouts.dtype != torch.float32:
#                 batch_gt_2d_layouts = batch_gt_2d_layouts.float()

#             # 禁用 autocast
#             with torch.cuda.amp.autocast(enabled=False):
#                 _, output_layout = navi(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1))
#                 loss_layout = criterion_layout(output_layout, batch_gt_2d_layouts.to(device))
#         else:
#             loss_layout = torch.tensor(0.0, device=device)

#         # 计算总损失
#         batch_loss = loss_maps + loss_layout
        
#         # 使用 GradScaler 处理反向传播
#         scaler.scale(batch_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         training_loss.append(batch_loss.item())

#     return training_loss

def validation_model(training_set_db, params, navi, device):
    criterion_maps = nn.MSELoss()
    criterion_layout = nn.BCELoss()

    accumulated_loss = 0
    update_count = 0

    # 遍历数据集，分批处理
    for i in range(0, len(training_set_db), params.navi_batch_size):
        batch_data = training_set_db[i:i + params.navi_batch_size]
        training_exp = []
        for data in batch_data:
            current_model_input = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
            current_gt_2d_layout = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)
            target_heatmap_pixel = torch.from_numpy(np.copy(data['target_heatmap_pixel'])).to(device)
            actual_path_gain = torch.from_numpy(np.copy(data['actual_path_gain'])).to(device)
            training_exp.append([
                current_model_input,
                current_gt_2d_layout,
                target_heatmap_pixel,
                actual_path_gain
            ])

        # 将数据加载到 GPU
        input_images = torch.cat([data[0] for data in training_exp])
        layouts = torch.cat([data[1] for data in training_exp])
        coords_list = [data[2] for data in training_exp]
        gt_pixels_list = [data[3] for data in training_exp]

        # 为 coords 和 gt_pixels 创建批处理索引
        batch_coords = torch.cat(coords_list).to(device)
        batch_gt_pixels = torch.cat(gt_pixels_list).to(device)
        batch_sizes = [len(coords) for coords in coords_list]
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), torch.tensor(batch_sizes, device=device))

        pred_heatmaps, pred_layouts = navi(input_images)
        pred_values = pred_heatmaps[batch_indices, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]]

        # 计算损失
        heatmaps_loss = criterion_maps(pred_values, batch_gt_pixels)
        layout_loss = criterion_layout(pred_layouts, layouts)
        batch_loss = heatmaps_loss + layout_loss

        accumulated_loss += batch_loss.item()
        update_count += 1

    validation_loss = accumulated_loss / update_count
    print("Validation loss:", validation_loss)

    return validation_loss

def train_experience_data(training_set_db, params, optimizer, navi, device, current_epoch):
    random.shuffle(training_set_db)
    # criterion_maps = nn.MSELoss()
    # criterion_layout = nn.BCELoss()
    scaler = GradScaler()
    
    # # 预处理所有数据并转移到GPU
    # input_images = torch.cat([data[0] for data in training_exp]).to(device)
    # layouts = torch.cat([data[1] for data in training_exp]).to(device)
    # coords_list = [data[2] for data in training_exp]
    # gt_pixels_list = [data[3] for data in training_exp]

    # # 创建全局索引
    # batch_starts = range(0, len(training_exp), params.navi_batch_size)
    # batch_ends = [min(start + params.navi_batch_size, len(training_exp)) for start in batch_starts]

    training_loss = []
    accumulated_loss = 0
    accumulation_steps = 8 # gaile: 8
    update_count = 0

    # 遍历数据集，分批处理
    for i in range(0, len(training_set_db), params.navi_batch_size):
        batch_data = training_set_db[i:i + params.navi_batch_size]
        training_exp = []
        for data in batch_data:
            if (data['pose_i'] > 10 and current_epoch == 1) or current_epoch > 1:  # current_epoch > 1
                current_model_input = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
                current_gt_2d_layout = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)
                target_heatmap_pixel = torch.from_numpy(np.copy(data['target_heatmap_pixel'])).to(device)
                actual_path_gain = torch.from_numpy(np.copy(data['actual_path_gain'])).to(device)
                # if isinstance(data['current_model_input'], torch.Tensor):
                #     data['current_model_input'] = data['current_model_input']
                # else: 
                #     data['current_model_input'] = torch.from_numpy(np.copy(data['current_model_input'])).to(device)

                # if isinstance(data['current_gt_2d_layout'], torch.Tensor):
                #     data['current_gt_2d_layout'] = data['current_gt_2d_layout']
                # else: 
                #     data['current_gt_2d_layout'] = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)

                # if isinstance(data['target_heatmap_pixel'], torch.Tensor):
                #     data['target_heatmap_pixel'] = data['target_heatmap_pixel']
                # else:
                #     data['target_heatmap_pixel'] = torch.from_numpy(np.copy(data['target_heatmap_pixel'])).to(device)

                # if isinstance(data['actual_path_gain'], torch.Tensor):
                #     data['actual_path_gain'] = data['actual_path_gain']
                # else:
                #     data['actual_path_gain'] = torch.from_numpy(np.copy(data['actual_path_gain'])).to(device)
                training_exp.append([
                    current_model_input,
                    current_gt_2d_layout,
                    target_heatmap_pixel,
                    actual_path_gain
                ])
        
        if not training_exp:
            continue

        # 将数据加载到 GPU
        input_images = torch.cat([data[0] for data in training_exp])
        layouts = torch.cat([data[1] for data in training_exp])
        coords_list = [data[2] for data in training_exp]
        gt_pixels_list = [data[3] for data in training_exp]

        # 为 coords 和 gt_pixels 创建批处理索引
        batch_coords = torch.cat(coords_list).to(device)
        batch_gt_pixels = torch.cat(gt_pixels_list).to(device)
        batch_sizes = [len(coords) for coords in coords_list]
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), torch.tensor(batch_sizes, device=device))

        pred_heatmaps, pred_layouts = navi(input_images)
        pred_values = pred_heatmaps[batch_indices, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]]

        # # 计算损失
        # heatmaps_loss = criterion_maps(pred_values, batch_gt_pixels)
        # layout_loss = criterion_layout(pred_layouts, layouts)
        # batch_loss = heatmaps_loss + layout_loss * 100
        batch_loss = navi.loss(pred_values, batch_gt_pixels, pred_layouts, layouts)

        # 反向传播
        scaler.scale(batch_loss).backward()
        accumulated_loss += batch_loss.item()
        update_count += 1

        # 参数更新
        if update_count % accumulation_steps == 0 or (i + params.navi_batch_size) >= len(training_set_db):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            training_loss.append(accumulated_loss / accumulation_steps)
            accumulated_loss = 0
            update_count = 0

    return training_loss

def evaluate_experience_data(all_experiences, params, navi, average_kernel, kernel_size, device):
    validation_losses = []
    total_batches = len(all_experiences) // params.navi_batch_size + (1 if len(all_experiences) % params.navi_batch_size > 0 else 0)
    criterion_maps = nn.MSELoss()  # MSE用于像素点预测
    criterion_layout = nn.BCELoss()  # BCE用于经过Sigmoid的整图预测
    random.shuffle(all_experiences)
    for batch_index in range(total_batches):
        start_index = batch_index * params.navi_batch_size
        end_index = start_index + params.navi_batch_size
        experiences = all_experiences[start_index:end_index]

        batch_start_grids, batch_previous_trajectories, batch_gt_2d_layouts, batch_target_locations, batch_angles, batch_gains = zip(*experiences)
        batch_start_grids = torch.cat(batch_start_grids).to(device)
        batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
        batch_gt_2d_layouts = torch.cat(batch_gt_2d_layouts).to(device)
        batch_target_locations = torch.cat(batch_target_locations).to(device)
        batch_angles = torch.cat(batch_angles).to(device)
        batch_gains = torch.tensor(batch_gains, dtype=torch.float32, device=device)

        output_maps = navi(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1).to(device))

        # batch_loss = torch.mean(torch.stack(mse_losses))
        indices = torch.arange(len(batch_angles), device=device)
        predicted_potentials = output_maps[indices, batch_angles, batch_target_locations[:, 0], batch_target_locations[:, 1]]
        loss_maps = criterion_maps(predicted_potentials, batch_gains)
        # loss_layout = criterion_layout(output_layout, batch_gt_2d_layouts) 
        batch_loss = loss_maps # + loss_layout 
        validation_losses.append(batch_loss.item())

    return np.mean(validation_losses)


#---------Functons to train the model------------
def train_navi(db_env, params, optimizer, navi, device, folder_img_path, current_epoch, validation_data, last_experience_num=2304, validation_split=0.1, patience=10, lr_patience=2, lr_factor=0.1):
    
    train_losses = []
    validation_losses = []

    # if len(navi_memory_buffer.buffer) > 0:
    #     all_experiences = list(navi_memory_buffer.buffer)
    #     if last_experience_num > 0:
    #         old_experience  = all_experiences[0: last_experience_num]
    #         new_experience = all_experiences[last_experience_num:] # used for training heatmap
    #         training_exp = new_experience + random.sample(old_experience, len(old_experience) // 10)
    #     else:
    #         training_exp= all_experiences
    #     num_epochs= 3

    #     random.shuffle(training_exp)
    if True: 
        if current_epoch == 1:
            last_experience_num = None
            training_set_db = read_combined_data(db_env, sample_m=last_experience_num)
        else:
            training_set_db = read_combined_data(db_env)
            print(len(training_set_db))
        # training_exp = []
        # for data in training_set_db:
        #     if (data['pose_i'] > 10 and current_epoch == 1) or current_epoch > 1:
        #         data['current_model_input'] = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
        #         data['current_gt_2d_layout'] = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)
        #         data['target_heatmap_pixel'] = torch.from_numpy(np.copy(data['target_heatmap_pixel'])).to(device)
        #         data['actual_path_gain'] = torch.from_numpy(np.copy(data['actual_path_gain'])).to(device)
        #         training_exp.append([data['current_model_input'], data['current_gt_2d_layout'], data['target_heatmap_pixel'], data['actual_path_gain']])

        num_epochs = 5
        # split_idx = int(len(all_experiences) * (1 - validation_split))
        # training_experiences, validation_experiences = all_experiences[:split_idx], all_experiences[split_idx:]

        # early_stopping = EarlyStopping(patience=patience)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True)

        for epoch in range(num_epochs):
            navi.train()
            
            train_loss = train_experience_data(training_set_db=training_set_db, params=params, optimizer=optimizer, navi=navi, device=device, current_epoch=current_epoch)
            navi.eval()
            with torch.no_grad():
                validation_loss = validation_model(validation_data, params, navi, device)
            # navi.eval()
            # with torch.no_grad():
            #     validation_loss = evaluate_experience_data(all_experiences=validation_experiences, params=params, navi=navi, average_kernel=average_kernel, kernel_size=kernel_size, device=device)
            print(np.mean(train_loss))
            # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {np.mean(train_loss)}, Validation Loss: {validation_loss}")
            train_losses.append(np.mean(train_loss))
            validation_losses.append((validation_loss))
            # val_losses.append(validation_loss)
            # early_stopping(validation_loss)
            lr_scheduler.step((validation_loss))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}")

            # if early_stopping.early_stop:
            #     print("Early stopping triggered.")
            #     return validation_loss
    average_loss = sum(train_losses) / len(train_losses)
    average_validation_loss = sum(validation_losses) / len(validation_losses)
    del training_set_db
    gc.collect()
    return average_loss, average_validation_loss

def trajectory_collection(params, current_epoch, train_dataloader, db_env, grid_size, heatmap_grid_size, grid_range, kernel_size, average_kernel,
                        navi, coverage_after_trajectory, macarons, memory, device, folder_img_path):
    
    use_perfect_depth_map = True
    num_scenes = 0
    for batch, scene_dict in enumerate(train_dataloader):
        if (num_scenes+1) % 35 == 0:
            print(num_scenes)
        num_scenes += 1
        
        scene_names = scene_dict['scene_name']
        obj_names = scene_dict['obj_name']
        all_settings = scene_dict['settings']
        # occupied_pose_datas = scene_dict['occupied_pose']
        batch_size = 1
        
        for i_scene in range(batch_size):
            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            # occupied_pose_data = occupied_pose_datas[i_scene]
            # print("\nScene name:", scene_name)
            # print("-------------------------------------")
            
            scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            
            #---ToChange: mirror scenes
            mirrored_scene = False # (bool) If True, mirrors the coordinates on x-axis
            mirrored_axis = None
            
            mesh = load_scene_with_texture(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)

            # split pc to n pieces
            n_pieces = 4
            verts = mesh.verts_list()[0]
            min_y, max_y = torch.min(verts, dim=0)[0][1].item() + 0.5, torch.max(verts, dim=0)[0][1].item() - 0.5

            bin_width = (max_y - min_y) / n_pieces
            y_bins = torch.arange(min_y, max_y+bin_width, bin_width, device=device)


            # store a mesh via trimesh for collision check
            mesh_for_check = trimesh.load(mesh_path)
            mesh_for_check.vertices *= params.scene_scale_factor
            
            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
            
            gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
            test_resolution = 0.05
            gc.collect()
            torch.cuda.empty_cache()
            gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                    mesh,
                                                                                    settings,
                                                                                    mirrored_scene,
                                                                                    device,
                                                                                    mirrored_axis=None,
                                                                                    surface_scene_feature_dim=3,
                                                                                    test_resolution=test_resolution)
            
            start_cam_idx = settings.camera.start_positions[0]
            
            # sstup_training_2d_camera
            occupied_pose_data = None
            camera = setup_test_2d_camera(params, mesh, mesh_for_check, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
            
            # print(camera.X_cam_history[0], camera.V_cam_history[0])
            
            macarons.eval()

            full_pc = torch.zeros(0, 3, device=device)
            full_pc_colors = torch.zeros(0, 3, device=device)

            coverage_evolution = []
            
            # occupancy_X = None
            # occupancy_sigma = None

            Dijkstra_path = []
            path_record = 0
            unreachable_position = []
            
            # Split camera dictionary to: {key, (first_tensor, second_tensor)}
            splited_pose_space_idx = camera.generate_new_splited_dict()
            
            # splited_pose_space: '[8,  0,  8]': tensor([43.5556,  0.0000, 93.5556], device='cuda:1')
            splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

            # compute the bounding box for the gt_pc
            gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)
            
            
            # tensor([-4.3675e+01,  7.8750e-03, -5.3303e+01], device='cuda:1')
            # tensor([58.9199,  5.1122, 29.2644], device='cuda:1')
            
            experiences_list = []
            
            for pose_i in range(100):
                if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
                    fill_surface_scene(surface_scene, full_pc,
                                    random_sampling_max_size=params.n_gt_surface_points,
                                    min_n_points_per_cell_fill=3,
                                    progressive_fill=params.progressive_fill,
                                    max_n_points_per_fill=params.max_points_per_progressive_fill,
                                    full_pc_colors=full_pc_colors)
            
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                                n_frames=1,
                                                                                                n_alpha=params.n_alpha,
                                                                                                return_gt_zbuf=True)                          
                                                                                                
            
                # for i in range(all_zbuf[-1:].shape[0]):
                # # TO CHANGE: filter points based on SSIM value!
                #     part_pc = camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                #                                                 mask=all_mask[-1:],
                #                                                 fov_cameras=camera.get_fov_camera_from_RT(
                #                                                     R_cam=all_R[-1:],
                #                                                     T_cam=all_T[-1:]),
                #                                                 gathering_factor=params.gathering_factor,
                #                                                 fov_range=params.sensor_range)

                #     # Fill surface scene
                #     part_pc_features = torch.zeros(len(part_pc), 1, device=device)
                #     covered_scene.fill_cells(part_pc, features=part_pc_features)  # Todo: change for better surface
                    # Compute true coverage for evaluation
                # current_coverage = gt_scene.scene_coverage(covered_scene,
                #                                         surface_epsilon=2.5 * test_resolution * params.scene_scale_factor)
                current_coverage = calculate_similarity_pcs(gt_scene_pc, full_pc)

                # if pose_i % 10 == 0:
                #     print("current coverage:", current_coverage)
                
                # if current_coverage[0] == 0.:
                #     coverage_evolution.append(0.)
                # else:
                #     coverage_evolution.append(current_coverage[0].item())
                coverage_evolution.append(current_coverage)
                    
                if pose_i == params.n_poses_in_trajectory:
                    coverage_after_trajectory.append(current_coverage)

                if current_coverage > 0.95:
                    break
                        
                torch.cuda.empty_cache()
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                                all_images=all_images, all_mask=all_mask,
                                                                all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                                mode='inference', device=device,
                                                                all_zbuf=all_zbuf)

                # Depth prediction
                with torch.no_grad():
                    depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=params,
                                                                            macarons=macarons.depth,
                                                                            batch_dict=batch_dict,
                                                                            alpha_dict=alpha_dict,
                                                                            device=device,
                                                                            use_perfect_depth=params.use_perfect_depth)
                    
                
                if use_perfect_depth_map:
                    depth = all_zbuf[2:3]
                    error_mask = mask

                # We fill the surface scene with the partial point cloud
                for i in range(depth.shape[0]):
                    # TO CHANGE: filter points based on SSIM value!
                    part_pc, part_pc_features = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                images=batch_dict["images"][i:i+1],
                                                                mask=(mask * error_mask)[i:i + 1],
                                                                fov_cameras=camera.get_fov_camera_from_RT(
                                                                    R_cam=batch_dict['R'][i:i + 1],
                                                                    T_cam=batch_dict['T'][i:i + 1]),
                                                                gathering_factor=params.gathering_factor,
                                                                fov_range=params.sensor_range)

                    # Fill surface scene
                    # part_pc_features = torch.zeros(len(part_pc), 1, device=device)
                    surface_scene.fill_cells(part_pc, features=part_pc_features)

                    full_pc = torch.vstack((full_pc, part_pc))
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_features))
                

                # ----------Update Proxy Points data with current FoV-----------------------------------------------------------

                # # Get Proxy Points in current FoV
                # fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, 
                #                                                             return_mask=True,
                #                                                             fov_camera=None,
                #                                                             fov_range=params.sensor_range)
                
                # fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
                # proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

                # # Computing signed distance of proxy points in fov
                # sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                #                                                     depth_maps=depth,
                #                                                     mask=mask, fov_camera=None)

                # # Updating view_state vectors
                # proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
                #                                     signed_distances=sgn_dists,
                #                                     distance_to_surface=None, X_cam=None)  # distance_to_surface TO CHANGE!

                # # Update the supervision occupancy for proxy points using the signed distance
                # proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

                # # Update the out-of-field status for proxy points inside camera field of view
                # proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

                # Update visibility history of surface points
                # surface_scene.set_all_features_to_value(value=1.)

                # ----------Predict Occupancy Probability Field-----------------------------------------------------------------
                
                # with torch.no_grad():
                #     X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone,
                #                                                                                 camera,
                #                                                                                 surface_scene, proxy_scene,
                #                                                                                 device)
                # occupancy_X = X_world + 0
                # occupancy_sigma = occ_probs + 0
                # occupancy_sigma = torch.clamp(occupancy_sigma, min=0, max=1)

                camera_current_pose, _ = camera.get_pose_from_idx(camera.cam_idx)

                #---------------------------------------------------------------------------------------------------------
                # Data collection
                #---------------------------------------------------------------------------------------------------------
                # surface pc collection
                # We first divvide current full_pc into n pieces
                bins = torch.bucketize(full_pc[:, 1], y_bins[:-1]) - 1 # compute the range of n_pieces: exact the index of y_values
                full_pc_groups = [full_pc[bins == i] for i in range(n_pieces)]
                full_pc_images = []

                for i in range(n_pieces):
                    if len(full_pc_groups[i]) > 0:
                        points_2d_batch = batch_transform_points_n_pieces(full_pc_groups[i], camera_current_pose, device)
                        current_partial_pc_img = map_points_to_grid_n_pieces(points_2d_batch, grid_size, grid_range, device) # size(1, 100, 100)
                    else:
                        current_partial_pc_img = torch.zeros(1, grid_size[0], grid_size[1], device=device)
                    full_pc_images.append(current_partial_pc_img)

                full_pc_images = torch.cat(full_pc_images, dim=0)
                navi_input_current_img = full_pc_images.unsqueeze(0)

                # gt_layout collection
                occ_array = get_binary_layout_array(mesh_for_check, camera_current_pose, view_size=grid_range[1]*2)
                current_gt_occupancy = torch.tensor(occ_array, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # previsous camera trajectory collection:
                trajectory_2d = batch_transform_points_n_pieces(camera.X_cam_history, camera_current_pose, device)
                previous_trajectory_img = map_points_to_grid_n_pieces(trajectory_2d, grid_size, grid_range, device)
                current_previous_trajectory_img = previous_trajectory_img.unsqueeze(0)

                ############################################################################################################
           

                if path_record + 1 > len(Dijkstra_path):
                    
                    # Data collection for valude model (last trajectory):
                    if len(experiences_list) > 0:
                        for ex in range(len(experiences_list)):
                            # apply data augmentation: search a pixels in a single map
                            if ex+1 <= len(experiences_list):
                                pixel_list = []
                                gain_list = []
                                for ex_next in range(ex+1, len(experiences_list)):
                                    ex_next_location = batch_transform_points_n_pieces(experiences_list[ex_next][-2][:3].unsqueeze(0), experiences_list[ex][-2], device)
                                    ex_grid_position = get_grid_position(ex_next_location.squeeze(0), heatmap_grid_size, grid_range)
                                    if 0 <= ex_grid_position[0] < heatmap_grid_size[0] and 0 <= ex_grid_position[1] < heatmap_grid_size[1]:
                                        ex_cam_img_position = torch.tensor([ex_grid_position[0], ex_grid_position[1]]).to(device)
                                        # Avoid actual gain smaller than 0
                                        actual_gain = (experiences_list[ex_next][0]-experiences_list[ex][0])*100 if (experiences_list[ex_next][0]-experiences_list[ex][0]) > 0 else 0

                                        current_pixel = torch.cat((experiences_list[ex_next][-1].unsqueeze(0), ex_cam_img_position), dim=0)
                                        pixel_list.append(current_pixel)
                                        gain_list.append(actual_gain)
                                if len(pixel_list) > 0: 
                                    pixels = torch.stack(pixel_list, dim=0)
                                    stack_gains = torch.tensor(gain_list, dtype=torch.float32, device=device)
                                    # navi_memory_buffer.add(Experience(current_model_input=experiences_list[ex][1], current_gt_2d_layout=experiences_list[ex][2],
                                    #                                 target_heatmap_pixel=pixels,
                                    #                              actual_path_gain=stack_gains))
                                    experience_db = {
                                        'current_model_input': experiences_list[ex][1],
                                        'current_gt_2d_layout': experiences_list[ex][2],
                                        'target_heatmap_pixel': pixels,
                                        'actual_path_gain': stack_gains,
                                        'pose_i' : pose_i
                                    }
                                    store_experience(db_env, experience_db)


                                    
                        experiences_list = []    
                    
                    # if len(experiences_list) > 0:   
                    #     total_experience.append(experiences_list) # Store a single dijkstra data to total data
                    #     experiences_list = []
                    # Dijkstra_gain_log.append([coverage_evolution[-1], random_flag])
                    # if len(Dijkstra_gain_log) >= 3 and (not Dijkstra_gain_log[-1][1]):
                    #     # one_path_gain = (Dijkstra_gain_log[-1][0] - Dijkstra_gain_log[-2][0]) * 100
                    #     one_path_exp = total_experience[len(Dijkstra_gain_log)-3]
                        
                    #     for ex in range(len(one_path_exp)-1):
                    #         ex_target_location = batch_transform_points(one_path_exp[-1][-2][:3].unsqueeze(0), one_path_exp[ex][-2].unsqueeze(0), device)
                    #         ex_grid_location = get_grid_position(ex_target_location.squeeze(0), grid_size, grid_range)
                    #         if 0 <= ex_grid_location[0] < grid_size[0] and 0 <= ex_grid_location[1] < grid_size[1]:
                    #             ex_target_position = torch.tensor([grid_size[0]-ex_grid_location[1].item()-1, ex_grid_location[0].item()]).to(device)
                    #             actual_gain = (Dijkstra_gain_log[-1][0]-one_path_exp[ex][0])*100 if (Dijkstra_gain_log[-1][0]-one_path_exp[ex][0]) > 0 else 0
                    #             navi_memory_buffer.add(Experience(start_pose_grid_map=one_path_exp[ex][1], target_pose_location=ex_target_position.unsqueeze(0), 
                    #                                               camera_angle_idx=one_path_exp[ex][-1].unsqueeze(0),
                    #                                                        actual_path_gain=actual_gain))
                        
                    Dijkstra_path = []
                    path_record = 0
                    
                    # # Generate occupancy 2d maps
                    # points_2d_batch = batch_transform_points_optimized(occupancy_X, camera_current_pose, device)
                    # current_camera_grids = map_points_to_grid_optimized(points_2d_batch, occupancy_sigma, grid_size, grid_range) # size(1, 100, 100)
                    # Generate full_pc 2d maps

                    # 使用模型进行批量增益预测 [92, 1, 64, 64] [92, 1, 64, 64] [92, 1]

                    current_model_input = torch.cat((navi_input_current_img, current_previous_trajectory_img), dim=1).to(device)
                    gain_map_prediction, _ = navi(current_model_input)

                    # mean_gain_map = torch.mean(gain_map_prediction, dim=1, keepdim=True)
                    max_gain_map, _ = torch.max(gain_map_prediction, dim=1, keepdim=True)
                    camera_position_value_list = []
                    for key, point_3d in splited_pose_space.items():
                        if camera.cam_idx[:3].tolist() != ast.literal_eval(key):
                            point_2d = batch_transform_points_n_pieces(point_3d.unsqueeze(0), camera_current_pose, device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')
                            grid_position = get_grid_position(point_2d.squeeze(0), heatmap_grid_size, grid_range)
                            
                            if 0 <= grid_position[0] < heatmap_grid_size[0] and 0 <= grid_position[1] < heatmap_grid_size[1]:
                                # Change to locations in image: (H - 1 - y, x)
                                cam_img_position = torch.tensor([grid_position[0], grid_position[1]]).to(device)

                                # average_result = F.conv2d(max_gain_map, average_kernel, padding=(kernel_size//2, kernel_size//2))
                                average_cam_img_result = max_gain_map[0, 0, cam_img_position[0], cam_img_position[1]].detach()
                                # Store these info to a list
                                new_list_grid_position = []
                                new_list_grid_position.append(key)
                                new_list_grid_position.append(cam_img_position)
                                new_list_grid_position.append(average_cam_img_result)
                                camera_position_value_list.append(new_list_grid_position)
                            
                    if random.random() <= 0.7:
                        camera_position_value_list.sort(key=lambda x: x[-1].item(), reverse=True)
                    else:
                        random.shuffle(camera_position_value_list)

                    path_start_position = camera.cam_idx[:3].tolist()
                    for pose_location in camera_position_value_list:

                        path_end_position = ast.literal_eval(pose_location[0])

                        # ray_directions = np.array([[0, -1, 0]])
                        # inter_locations, _, _ = mesh_for_check.ray.intersects_location(ray_origins=[splited_pose_space[pose_location[0]].cpu().numpy()],
                        #                                                                 ray_directions=ray_directions)
                        
                        # if len(inter_locations) % 2 == 1:
                        if check_camera_in_mesh(mesh_for_check, splited_pose_space[pose_location[0]]):

                            if path_end_position in unreachable_position:
                                continue
                            
                            Dijkstra_path = generate_Dijkstra_path_2d(splited_pose_space, path_start_position, path_end_position, mesh_for_check, camera_current_pose, camera,
                                                                    heatmap_grid_size, grid_range, gain_map_prediction, device)
                            if Dijkstra_path is not None:
                                experiences_list.append([
                                        coverage_evolution[-1],
                                        current_model_input,
                                        current_gt_occupancy,
                                        camera_current_pose,
                                        camera.cam_idx[-1]
                                    ])
                                break
                            else:
                                unreachable_position.append(path_end_position)
                            
                else:
                    experiences_list.append([
                                        coverage_evolution[-1],
                                        current_model_input,
                                        current_gt_occupancy,
                                        camera_current_pose,
                                        camera.cam_idx[-1]
                                    ])

                    # # 使用模型进行批量增益预测 [92, 1, 64, 64] [92, 1, 64, 64] [92, 1]
                    # gain_map_prediction = navi(navi_input_current_img)

                    # camera_position_value_list = []
                    # for key, point_3d in splited_pose_space.items():
                    #     point_2d = batch_transform_points(point_3d.unsqueeze(0), camera_current_pose.unsqueeze(0), device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')
                    #     grid_position = get_grid_position(point_2d.squeeze(0), grid_size, grid_range)
                        
                    #     if 0 <= grid_position[0] < grid_size[0] and 0 <= grid_position[1] < grid_size[1]:
                    #         # Change to locations in image: (H - 1 - y, x)
                    #         cam_img_position = torch.tensor([grid_size[0]-grid_position[1].item()-1, grid_position[0].item()]).to(device)

                    #         average_result = F.conv2d(gain_map_prediction, average_kernel, padding=(kernel_size//2, kernel_size//2))
                    #         average_cam_img_result = average_result[0, 0, cam_img_position[0], cam_img_position[1]].detach()
                            
                    #         # Store these info to a list
                    #         new_list_grid_position = []
                    #         new_list_grid_position.append(key)
                    #         new_list_grid_position.append(cam_img_position)
                    #         new_list_grid_position.append(average_cam_img_result)
                    #         camera_position_value_list.append(new_list_grid_position)
                    
                    # camera_position_value_list.sort(key=lambda x: x[-1].item(), reverse=True)

                    # for pose_location in camera_position_value_list:

                    #     path_start_position = camera.cam_idx[:3].tolist()
                    #     path_end_position = ast.literal_eval(pose_location[0])

                    #     if path_end_position in unreachable_position:
                    #         continue

                    #     fake_Dijkstra_path = generate_Dijkstra_path_2d(splited_pose_space, path_start_position, path_end_position, gt_scene, device)
                    #     if fake_Dijkstra_path:
                    #         current_exp_list = []
                    #         if pose_location[2].item() > max_average_cam_img_result:
                    #             Dijkstra_path = fake_Dijkstra_path
                    #             max_average_cam_img_result = pose_location[2].item()
                    #             path_record = 0
                    #             current_dij_target = splited_pose_space[pose_location[0]] 
                                
                    #             current_exp_list.append(navi_input_current_img)
                    #             current_exp_list.append(pose_location[1])
                    #         else:
                    #             point_2d_target = batch_transform_points(current_dij_target.unsqueeze(0), camera_current_pose.unsqueeze(0), device)
                    #             grid_target_position = get_grid_position(point_2d_target.squeeze(0), grid_size, grid_range)
                    #             if 0 <= grid_target_position[0] < grid_size[0] and 0 <= grid_target_position[1] < grid_size[1]:
                    #                 cam_img_target_position = torch.tensor([grid_size[0]-grid_target_position[1].item()-1, grid_target_position[0].item()]).to(device)
                    #                 current_exp_list.append(navi_input_current_img)
                    #                 current_exp_list.append(cam_img_target_position)
                    #         if len(current_exp_list) > 0:
                    #             experiences_list.append(current_exp_list)
                            
                    #         break
                                
                    #     else:
                    #         unreachable_position.append(path_end_position)
                    

                if Dijkstra_path is None:
                    break
                
                #-------------------------------------------------------------------
                # ------ Decide the specific camera pose along the Dijkstra path----
                #-------------------------------------------------------------------
                # neighbor_indices = camera.get_neighboring_poses_2d()
                # valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
                # dij_next_position = torch.tensor(Dijkstra_path[path_record]).to(device)
                # valid_neighbors_dij_candidates = valid_neighbors[(valid_neighbors[:, :3] == dij_next_position).all(dim=1)]
                
                # if valid_neighbors_dij_candidates.shape[0] == 0:
                #     valid_neighbors_dij_candidates = neighbor_indices[(neighbor_indices[:, :3] == dij_next_position).all(dim=1)]
                # max_coverage_gain = -1.
                # next_idx = None
                
                # # Use converage gain model to optimize neighboring poses
                # for neighbor_i in range(len(valid_neighbors_dij_candidates)):
                #     neighbor_idx = valid_neighbors_dij_candidates[neighbor_i]
                #     neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
                #     X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    
                #     with torch.no_grad():
                #         _, _,fov_proxy_volume, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                #             params=params, macarons=macarons.scone,
                #             proxy_scene=proxy_scene, surface_scene=surface_scene,
                #             X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                #             camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)
                #     if coverage_gain.shape[0] > 0:
                #         if coverage_gain.item() > max_coverage_gain:
                #             max_coverage_gain = coverage_gain.item()
                #             next_idx = neighbor_idx
                    
                # if next_idx is None:
                    
                #     next_idx = valid_neighbors_dij_candidates[torch.randint(0, valid_neighbors_dij_candidates.size(0), (1,)).item()]

                if path_record >= len(Dijkstra_path):
                    print(f"IndexError: Index {path_record} out of range for Dijkstra_path with length {len(Dijkstra_path)}.")
                    break

                next_idx = Dijkstra_path[path_record]
                if random.random() <= 0.6:
                    next_idx[-1] = torch.randint(0, 8, (1,))

                # Move to next camera pose
                interpolation_step = 1
                for i in range(camera.n_interpolation_steps):
                    camera.update_camera(next_idx, interpolation_step=interpolation_step)
                    camera.capture_image(mesh)
                    interpolation_step += 1
                    
                # Depth prediction
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
                                camera=camera,
                                n_frames=params.n_interpolation_steps,
                                n_alpha=params.n_alpha_for_supervision,
                                return_gt_zbuf=True)
                
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=device, all_zbuf=all_zbuf)
                # Depth prediction
                depth, mask, error_mask = [], [], []
                for i in range(batch_dict['images'].shape[0]):
                    batch_dict_i = {}
                    batch_dict_i['images'] = batch_dict['images'][i:i + 1]
                    batch_dict_i['mask'] = batch_dict['mask'][i:i + 1]
                    batch_dict_i['R'] = batch_dict['R'][i:i + 1]
                    batch_dict_i['T'] = batch_dict['T'][i:i + 1]
                    batch_dict_i['zfar'] = batch_dict['zfar'][i:i + 1]
                    batch_dict_i['zbuf'] = batch_dict['zbuf'][i:i + 1]

                    alpha_dict_i = {}
                    alpha_dict_i['images'] = alpha_dict['images'][i:i + 1]
                    alpha_dict_i['mask'] = alpha_dict['mask'][i:i + 1]
                    alpha_dict_i['R'] = alpha_dict['R'][i:i + 1]
                    alpha_dict_i['T'] = alpha_dict['T'][i:i + 1]
                    alpha_dict_i['zfar'] = alpha_dict['zfar'][i:i + 1]
                    alpha_dict_i['zbuf'] = alpha_dict['zbuf'][i:i + 1]
                    

                    with torch.no_grad():
                        depth_i, mask_i, error_mask_i, _, _ = apply_depth_model(params=params, macarons=macarons.depth,
                                                                                batch_dict=batch_dict_i,
                                                                                alpha_dict=alpha_dict_i,
                                                                                device=device,
                                                                                compute_loss=False,
                                                                                use_perfect_depth=params.use_perfect_depth)
                        if use_perfect_depth_map:
                            depth_i = all_zbuf[2+i:3+i]
                            error_mask_i = mask_i

                    depth.append(depth_i)
                    mask.append(mask_i)
                    error_mask.append(error_mask_i)


                depth = torch.cat(depth, dim=0)
                mask = torch.cat(mask, dim=0)
                error_mask = torch.cat(error_mask, dim=0)
                
                # ----------Build supervision signal from the new depth maps----------------------------------------------------
                all_part_pc = []
                all_part_pc_features = []
                # all_fov_proxy_points = torch.zeros(0, 3, device=device)
                # general_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()
                # all_fov_proxy_mask = []
                # all_sgn_dists = []
                all_X_cam = []
                all_fov_camera = []
                
                for i in range(depth.shape[0]):
                    fov_frame = camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
                    all_X_cam.append(fov_frame.get_camera_center())
                    all_fov_camera.append(fov_frame)

                    # TO CHANGE: filter points based on SSIM value!
                    part_pc, part_pc_features = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                images=batch_dict['images'][i:i+1],
                                                                mask=(mask * error_mask)[i:i + 1].bool(),
                                                                fov_cameras=fov_frame,
                                                                gathering_factor=params.gathering_factor,
                                                                fov_range=params.sensor_range)

                    # Surface points to fill surface scene
                    all_part_pc.append(part_pc)
                    all_part_pc_features.append(part_pc_features)

                    # # Get Proxy Points in current FoV
                    # fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                    #                                                             fov_camera=fov_frame, fov_range=params.sensor_range)
                    # all_fov_proxy_points = torch.vstack((all_fov_proxy_points, fov_proxy_points))
                    # all_fov_proxy_mask.append(fov_proxy_mask)
                    # general_fov_proxy_mask = general_fov_proxy_mask + fov_proxy_mask

                    # # Computing signed distance of proxy points in fov
                    # sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[i:i + 1],
                    #                                                     mask=mask[i:i + 1].bool(), fov_camera=fov_frame
                    #                                                     ).view(-1, 1)
                    # all_sgn_dists.append(sgn_dists)

                    # Computing mask for proxy points close to the surface.
                    # We will use this for occupancy probability supervision.
                    # close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

                # ----------Update Scenes to finalize supervision signal and prepare next iteration-----------------------------

                # 1. Surface scene
                # Fill surface scene
                # We give a visibility=1 to points that were visible in frame t, and 0 to others
                complete_part_pc = torch.vstack(all_part_pc)
                complete_part_pc_features = torch.vstack(all_part_pc_features)
                # complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
                # complete_part_pc_features[:len(all_part_pc[0])] = 1.
                surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

                full_pc = torch.vstack((full_pc, complete_part_pc))
                full_pc_colors = torch.vstack((full_pc_colors, complete_part_pc_features))

                # Compute coverage gain for each new camera pose
                # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
                # supervision_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
                # for i in range(depth.shape[0]):
                #     supervision_coverage_gains[i, 0] = surface_scene.camera_coverage_gain(all_part_pc[i],
                #                                                                           surface_epsilon=None)

                # # Update visibility history of surface points
                # surface_scene.set_all_features_to_value(value=1.)

                # 2. Proxy scene
                # Fill proxy scene
                # general_fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
                # proxy_scene.fill_cells(proxy_scene.proxy_points[general_fov_proxy_mask],
                #                     features=general_fov_proxy_indices.view(-1, 1))

                # for i in range(depth.shape[0]):
                #     # Updating view_state vectors
                #     proxy_scene.update_proxy_view_states(camera, all_fov_proxy_mask[i],
                #                                         signed_distances=all_sgn_dists[i],
                #                                         distance_to_surface=None,
                #                                         X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

                #     # Update the supervision occupancy for proxy points using the signed distance
                #     proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i],
                #                                             tol=params.carving_tolerance)

                # # Update the out-of-field status for proxy points inside camera field of view
                # proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)
                path_record += 1
                
                # if pose_i == params.n_poses_in_trajectory:

                #     #------plot occupancy map------
                #     camera_pose_draw = torch.tensor([1, 0, 50, 30, 90]).to(device)
                #     points_2d_batch_img = batch_transform_points(occupancy_X, camera_pose_draw.unsqueeze(0), device)
                #     current_camera_grids_batch_img = map_points_to_grid_optimized(points_2d_batch_img, occupancy_sigma, grid_size, grid_range)
                    
                #     flipped_after_transpose = torch.flip(current_camera_grids_batch_img.squeeze(0).permute(1, 0), [0])
                    
                #     plt.imshow(flipped_after_transpose.cpu().numpy(), cmap='nipy_spectral')
                #     plt.colorbar()
                #     plt.savefig(f'{folder_img_path}occ_map_{current_epoch}_{tra_num}.png')
                    
                #     plt.clf()
                    
                #     #------plot trajectory------
                    
                #     x = gt_scene_pc[:, 0].cpu().numpy()
                #     z = gt_scene_pc[:, 2].cpu().numpy()
                #     angle_adjustment = np.pi / 4
                #     plt.scatter(x, z, s=1, color='black')
                #     combined_camera_pose = torch.cat((camera.X_cam_history, camera.V_cam_history), dim=1)
                #     combined_camera_pose = combined_camera_pose.cpu().numpy()
                #     combined_camera_pose = combined_camera_pose[4:, :]
                    

                #     for i, pose in enumerate(combined_camera_pose):
                #         if i % 4 == 0:
                #             pose[3:] = np.deg2rad(pose[3:])
                            
                #             x, _, z, _, azimuth = pose
                #             plt.scatter(x, z, color='blue')  # 绘制相机位置
                #             # 计算并绘制两条射线
                #             line_length = 10  # 射线长度，可根据需要调整
                #             for angle in [angle_adjustment, -angle_adjustment]:
                #                 end_x = x + np.cos(azimuth + angle) * line_length
                #                 end_z = z + np.sin(azimuth + angle) * line_length
                #                 plt.plot([x, end_x], [z, end_z], color='purple')  # 绘制射线
                            
                #             if i > 0:  # 确保有前一个相机位置来绘制线段
                #                 prev_pose = combined_camera_pose[i - 4]  # 获取前一个相机的位置
                #                 prev_pose[3:] = np.deg2rad(prev_pose[3:])
                #                 prev_x, _, prev_z, _, _ = prev_pose
                #                 # 使用箭头指向当前相机位置
                #                 plt.arrow(prev_x, prev_z, x - prev_x, z - prev_z, color='red', head_width=2, length_includes_head=True)
                #     plt.title('Point Cloud XZ Plane View with Camera Poses')
                #     plt.xlabel('X axis')
                #     plt.ylabel('Z axis')
                #     plt.axis('equal')
                #     # 保存图像
                #     plt.savefig(f'{folder_img_path}pc_with_camera_poses_{current_epoch}_{tra_num}.png') 
                #     plt.clf()
                    
                #     #------plot heatmap---------
                            
                #     gain_map_prediction = navi(flipped_after_transpose.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                #     plt.imshow(gain_map_prediction.detach().cpu().numpy(), cmap='nipy_spectral')
                #     plt.colorbar()
                #     plt.savefig(f'{folder_img_path}heatmap_{current_epoch}_{tra_num}.png') 
                #     plt.clf()
                    
                #     #------plot results--------
                #     x = full_pc[:, 0].cpu().numpy()
                #     z = full_pc[:, 2].cpu().numpy()
                    
                #     # 绘制点云的XZ平面视图
                #     plt.scatter(x, z, s=1, color='black')
                    
                #     # 设置图表属性
                #     plt.title('Reconsructed Point Cloud XZ Plane View')
                #     plt.xlabel('X axis')
                #     plt.ylabel('Z axis')
                #     plt.axis('equal')
                    
                #     # 保存图像
                #     plt.savefig(f'{folder_img_path}result_{current_epoch}_{tra_num}.png')
                #     plt.clf()  # 清除当前图形

        #-------------------------------------------------------------------
        # ------ Store experience in memory and update the model -----------
        #-------------------------------------------------------------------
        # auc = coverage_evolution[-1] * 100
        # if len(experiences_list) > 0:     
        #     for item in experiences_list:   
        #         navi_memory_buffer.add(Experience(start_pose_grid_map=item[0], target_pose_location=item[1].unsqueeze(0), actual_path_gain=auc))











def trajectory_validation(params, current_epoch, train_dataloader, grid_size, grid_range, kernel_size, average_kernel,
                        navi, coverage_after_trajectory, macarons, memory, device, tra_num, folder_img_path):
    
    use_perfect_depth_map = True
    for batch, scene_dict in enumerate(train_dataloader):
        
        scene_names = scene_dict['scene_name']
        obj_names = scene_dict['obj_name']
        all_settings = scene_dict['settings']
        # occupied_pose_datas = scene_dict['occupied_pose']
        batch_size = 1
        
        for i_scene in range(batch_size):
            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            # occupied_pose_data = occupied_pose_datas[i_scene]
            # print("\nScene name:", scene_name)
            # print("-------------------------------------")
            
            scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            
            #---ToChange: mirror scenes
            mirrored_scene = False # (bool) If True, mirrors the coordinates on x-axis
            mirrored_axis = None
            
            mesh = load_scene_with_texture(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
            mesh_for_check = trimesh.load(mesh_path)
            mesh_for_check.vertices *= params.scene_scale_factor

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)

            gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
            test_resolution = 0.05
            gc.collect()
            torch.cuda.empty_cache()
            gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                    mesh,
                                                                                    settings,
                                                                                    mirrored_scene,
                                                                                    device,
                                                                                    mirrored_axis=None,
                                                                                    surface_scene_feature_dim=3,
                                                                                    test_resolution=test_resolution)
            
            start_cam_idx = settings.camera.start_positions[0]
            
            # sstup_training_2d_camera
            occupied_pose_data = None
            camera = setup_test_2d_camera(params, mesh, mesh_for_check, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
            
            # print(camera.X_cam_history[0], camera.V_cam_history[0])
            
            macarons.eval()

            full_pc = torch.zeros(0, 3, device=device)
            full_pc_colors = torch.zeros(0, 3, device=device)

            coverage_evolution = []
            
            occupancy_X = None
            occupancy_sigma = None                      

            Dijkstra_path = []
            path_record = 0
            unreachable_position = []
            
            # Split camera dictionary to: {key, (first_tensor, second_tensor)}
            splited_pose_space_idx = camera.generate_new_splited_dict()
            
            # splited_pose_space: '[8,  0,  8]': tensor([43.5556,  0.0000, 93.5556], device='cuda:1')
            splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

            # compute the bounding box for the gt_pc
            gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)
            
            # tensor([-4.3675e+01,  7.8750e-03, -5.3303e+01], device='cuda:1')
            # tensor([58.9199,  5.1122, 29.2644], device='cuda:1')
            max_average_cam_img_result = -100.
            
            for pose_i in range(params.n_poses_in_trajectory+1):

                if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
                    fill_surface_scene(surface_scene, full_pc,
                                    random_sampling_max_size=params.n_gt_surface_points,
                                    min_n_points_per_cell_fill=3,
                                    progressive_fill=params.progressive_fill,
                                    max_n_points_per_fill=params.max_points_per_progressive_fill,
                                    full_pc_colors=full_pc_colors)
            
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                                n_frames=1,
                                                                                                n_alpha=params.n_alpha,
                                                                                                return_gt_zbuf=True)                          
                                                                                                
            
                # for i in range(all_zbuf[-1:].shape[0]):
                # # TO CHANGE: filter points based on SSIM value!
                #     part_pc = camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                #                                                 mask=all_mask[-1:],
                #                                                 fov_cameras=camera.get_fov_camera_from_RT(
                #                                                     R_cam=all_R[-1:],
                #                                                     T_cam=all_T[-1:]),
                #                                                 gathering_factor=params.gathering_factor,
                #                                                 fov_range=params.sensor_range)

                #     # Fill surface scene
                #     part_pc_features = torch.zeros(len(part_pc), 1, device=device)
                #     covered_scene.fill_cells(part_pc, features=part_pc_features)  # Todo: change for better surface
                #     # Compute true coverage for evaluation
                # current_coverage = gt_scene.scene_coverage(covered_scene,
                #                                         surface_epsilon=2.5 * test_resolution * params.scene_scale_factor)
                # # if pose_i % 10 == 0:
                # #     print("current coverage:", current_coverage)
                
                # if current_coverage[0] == 0.:
                #     coverage_evolution.append(0.)
                # else:
                #     coverage_evolution.append(current_coverage[0].item())
                current_coverage = calculate_similarity_pcs(gt_scene_pc, full_pc)
                coverage_evolution.append(current_coverage)
                    
                if pose_i == params.n_poses_in_trajectory:
                    coverage_after_trajectory.append(current_coverage)
                        
                torch.cuda.empty_cache()
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                                all_images=all_images, all_mask=all_mask,
                                                                all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                                mode='inference', device=device,
                                                                all_zbuf=all_zbuf)

                # Depth prediction
                with torch.no_grad():
                    depth, mask, error_mask, pose, gt_pose = apply_depth_model(params=params,
                                                                            macarons=macarons.depth,
                                                                            batch_dict=batch_dict,
                                                                            alpha_dict=alpha_dict,
                                                                            device=device,
                                                                            use_perfect_depth=params.use_perfect_depth)
                    
                
                if use_perfect_depth_map:
                    depth = all_zbuf[2:3]
                    error_mask = mask

                # We fill the surface scene with the partial point cloud
                for i in range(depth.shape[0]):
                    # TO CHANGE: filter points based on SSIM value!
                    part_pc, part_pc_features = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                images=batch_dict["images"][i:i+1],
                                                                mask=(mask * error_mask)[i:i + 1],
                                                                fov_cameras=camera.get_fov_camera_from_RT(
                                                                    R_cam=batch_dict['R'][i:i + 1],
                                                                    T_cam=batch_dict['T'][i:i + 1]),
                                                                gathering_factor=params.gathering_factor,
                                                                fov_range=params.sensor_range)

                    # Fill surface scene
                    # part_pc_features = torch.zeros(len(part_pc), 1, device=device)
                    surface_scene.fill_cells(part_pc, features=part_pc_features)

                    full_pc = torch.vstack((full_pc, part_pc))
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_features))

                # ----------Update Proxy Points data with current FoV-----------------------------------------------------------

                # Get Proxy Points in current FoV
                fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, 
                                                                            return_mask=True,
                                                                            fov_camera=None,
                                                                            fov_range=params.sensor_range)
                
                fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
                proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

                # Computing signed distance of proxy points in fov
                sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                                    depth_maps=depth,
                                                                    mask=mask, fov_camera=None)

                # Updating view_state vectors
                proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
                                                    signed_distances=sgn_dists,
                                                    distance_to_surface=None, X_cam=None)  # distance_to_surface TO CHANGE!

                # Update the supervision occupancy for proxy points using the signed distance
                proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

                # Update the out-of-field status for proxy points inside camera field of view
                proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

                # Update visibility history of surface points
                # surface_scene.set_all_features_to_value(value=1.)

                # ----------Predict Occupancy Probability Field-----------------------------------------------------------------
                
                with torch.no_grad():
                    X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone,
                                                                                                camera,
                                                                                                surface_scene, proxy_scene,
                                                                                                device)
                occupancy_X = X_world + 0
                occupancy_sigma = occ_probs + 0
                occupancy_sigma = torch.clamp(occupancy_sigma, min=0, max=1)

                camera_current_pose, _ = camera.get_pose_from_idx(camera.cam_idx)
                
                if path_record + 1 > len(Dijkstra_path):
                                   
                    Dijkstra_path = []
                    path_record = 0
                    
                    # Generate occupancy 2d maps
                    points_2d_batch = batch_transform_points_n_pieces(occupancy_X, camera_current_pose, device)
                    current_camera_grids = map_points_to_grid_optimized(points_2d_batch, occupancy_sigma, grid_size, grid_range) # size(1, 100, 100)
                    
                    # Convert current_camera_grids to img viewed by camera (transpose and then flip)
                    current_camera_grids_img = current_camera_grids.squeeze(0)
                    
                    
                    # Convert grids_batchcovert to model input (N, 1, 128, 128)
                    navi_input_current_img = current_camera_grids_img.unsqueeze(0).unsqueeze(0)  # 增加一个通道维度

                    # 使用模型进行批量增益预测 [92, 1, 64, 64] [92, 1, 64, 64] [92, 1]
                    gain_map_prediction = navi(navi_input_current_img)
                    # mean_gain_map = torch.mean(gain_map_prediction, dim=1, keepdim=True)
                    max_gain_map, _ = torch.max(gain_map_prediction, dim=1, keepdim=True)

                    camera_position_value_list = []
                    for key, point_3d in splited_pose_space.items():
                        if camera.cam_idx[:3].tolist() != ast.literal_eval(key):
                            point_2d = batch_transform_points_n_pieces(point_3d.unsqueeze(0), camera_current_pose, device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')
                            grid_position = get_grid_position(point_2d.squeeze(0), grid_size, grid_range)
                            
                            if 0 <= grid_position[0] < grid_size[0] and 0 <= grid_position[1] < grid_size[1]:
                                # Change to locations in image: (H - 1 - y, x)
                                cam_img_position = torch.tensor([grid_size[0]-grid_position[1].item()-1, grid_position[0].item()]).to(device)

                                average_result = F.conv2d(max_gain_map, average_kernel, padding=(kernel_size//2, kernel_size//2))
                                average_cam_img_result = average_result[0, 0, cam_img_position[0], cam_img_position[1]].detach()
                                
                                # Store these info to a list
                                new_list_grid_position = []
                                new_list_grid_position.append(key)
                                new_list_grid_position.append(cam_img_position)
                                new_list_grid_position.append(average_cam_img_result)
                                camera_position_value_list.append(new_list_grid_position)
                            
                    camera_position_value_list.sort(key=lambda x: x[-1].item(), reverse=True)

                    for pose_location in camera_position_value_list:

                        path_start_position = camera.cam_idx[:3].tolist()
                        path_end_position = ast.literal_eval(pose_location[0])

                        if path_end_position in unreachable_position:
                            continue

                        Dijkstra_path = generate_Dijkstra_path_2d(splited_pose_space, path_start_position, path_end_position, gt_scene, 
                                                                  camera_current_pose, camera, 
                                                                  grid_size, grid_range, gain_map_prediction, device)

                        if len(Dijkstra_path) > 0:
    
                            # max_average_cam_img_result = pose_location[2].item()
                            break
                        else:
                            unreachable_position.append(path_end_position)
                else:
                    pass
                    # points_2d_batch = batch_transform_points(occupancy_X, camera_current_pose.unsqueeze(0), device)
                    # current_camera_grids = map_points_to_grid_optimized(points_2d_batch, occupancy_sigma, grid_size, grid_range) # size(1, 100, 100)
                    
                    # # Convert current_camera_grids to img viewed by camera (transpose and then flip)
                    # current_camera_grids_img = torch.flip(current_camera_grids.squeeze(0).permute(1, 0), [0])
                    
                    
                    # # Convert grids_batchcovert to model input (N, 1, 128, 128)
                    # navi_input_current_img = current_camera_grids_img.unsqueeze(0).unsqueeze(0)  # 增加一个通道维度


                    # # 使用模型进行批量增益预测 [92, 1, 64, 64] [92, 1, 64, 64] [92, 1]
                    # gain_map_prediction = navi(navi_input_current_img)

                    # camera_position_value_list = []
                    # for key, point_3d in splited_pose_space.items():
                    #     point_2d = batch_transform_points(point_3d.unsqueeze(0), camera_current_pose.unsqueeze(0), device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')
                    #     grid_position = get_grid_position(point_2d.squeeze(0), grid_size, grid_range)
                        
                    #     if 0 <= grid_position[0] < grid_size[0] and 0 <= grid_position[1] < grid_size[1]:
                    #         # Change to locations in image: (H - 1 - y, x)
                    #         cam_img_position = torch.tensor([grid_size[0]-grid_position[1].item()-1, grid_position[0].item()]).to(device)

                    #         average_result = F.conv2d(gain_map_prediction, average_kernel, padding=(kernel_size//2, kernel_size//2))
                    #         average_cam_img_result = average_result[0, 0, cam_img_position[0], cam_img_position[1]].detach()
                            
                    #         # Store these info to a list
                    #         new_list_grid_position = []
                    #         new_list_grid_position.append(key)
                    #         new_list_grid_position.append(cam_img_position)
                    #         new_list_grid_position.append(average_cam_img_result)
                    #         camera_position_value_list.append(new_list_grid_position)
                    
                    # camera_position_value_list.sort(key=lambda x: x[-1].item(), reverse=True)

                    # for pose_location in camera_position_value_list:

                    #     path_start_position = camera.cam_idx[:3].tolist()
                    #     path_end_position = ast.literal_eval(pose_location[0])

                    #     if path_end_position in unreachable_position:
                    #         continue
                    #     fake_Dijkstra_path = generate_Dijkstra_path_2d(splited_pose_space, path_start_position, path_end_position, gt_scene, device)
                    #     if fake_Dijkstra_path:
                    #         if pose_location[2].item() > max_average_cam_img_result:
                    #             Dijkstra_path = fake_Dijkstra_path
                    #             max_average_cam_img_result = pose_location[2].item()
                    #             path_record = 0
                    #     else:
                    #         unreachable_position.append(path_end_position)


                if len(Dijkstra_path) == 0:
                    break
                
                #-------------------------------------------------------------------
                # ------ Decide the specific camera pose along the Dijkstra path----
                #-------------------------------------------------------------------
                # neighbor_indices = camera.get_neighboring_poses_2d()
                # valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
                # dij_next_position = torch.tensor(Dijkstra_path[path_record]).to(device)
                # valid_neighbors_dij_candidates = valid_neighbors[(valid_neighbors[:, :3] == dij_next_position).all(dim=1)]
                
                # if valid_neighbors_dij_candidates.shape[0] == 0:
                #     valid_neighbors_dij_candidates = neighbor_indices[(neighbor_indices[:, :3] == dij_next_position).all(dim=1)]
                # max_coverage_gain = -1.
                # next_idx = None
                
                # # Use converage gain model to optimize neighboring poses
                # for neighbor_i in range(len(valid_neighbors_dij_candidates)):
                #     neighbor_idx = valid_neighbors_dij_candidates[neighbor_i]
                #     neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
                #     X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    
                #     with torch.no_grad():
                #         _, _,fov_proxy_volume, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                #             params=params, macarons=macarons.scone,
                #             proxy_scene=proxy_scene, surface_scene=surface_scene,
                #             X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                #             camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)
                #     if coverage_gain.shape[0] > 0:
                #         if coverage_gain.item() > max_coverage_gain:
                #             max_coverage_gain = coverage_gain.item()
                #             next_idx = neighbor_idx
                    
                # if next_idx is None:
                    
                #     next_idx = valid_neighbors_dij_candidates[torch.randint(0, valid_neighbors_dij_candidates.size(0), (1,)).item()]
                next_idx = Dijkstra_path[path_record]

                # Move to next camera pose
                interpolation_step = 1
                for i in range(camera.n_interpolation_steps):
                    camera.update_camera(next_idx, interpolation_step=interpolation_step)
                    camera.capture_image(mesh)
                    interpolation_step += 1
                    
                # Depth prediction
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
                                camera=camera,
                                n_frames=params.n_interpolation_steps,
                                n_alpha=params.n_alpha_for_supervision,
                                return_gt_zbuf=True)
                
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=device, all_zbuf=all_zbuf)
                # Depth prediction
                depth, mask, error_mask = [], [], []
                for i in range(batch_dict['images'].shape[0]):
                    batch_dict_i = {}
                    batch_dict_i['images'] = batch_dict['images'][i:i + 1]
                    batch_dict_i['mask'] = batch_dict['mask'][i:i + 1]
                    batch_dict_i['R'] = batch_dict['R'][i:i + 1]
                    batch_dict_i['T'] = batch_dict['T'][i:i + 1]
                    batch_dict_i['zfar'] = batch_dict['zfar'][i:i + 1]
                    batch_dict_i['zbuf'] = batch_dict['zbuf'][i:i + 1]

                    alpha_dict_i = {}
                    alpha_dict_i['images'] = alpha_dict['images'][i:i + 1]
                    alpha_dict_i['mask'] = alpha_dict['mask'][i:i + 1]
                    alpha_dict_i['R'] = alpha_dict['R'][i:i + 1]
                    alpha_dict_i['T'] = alpha_dict['T'][i:i + 1]
                    alpha_dict_i['zfar'] = alpha_dict['zfar'][i:i + 1]
                    alpha_dict_i['zbuf'] = alpha_dict['zbuf'][i:i + 1]
                    

                    with torch.no_grad():
                        depth_i, mask_i, error_mask_i, _, _ = apply_depth_model(params=params, macarons=macarons.depth,
                                                                                batch_dict=batch_dict_i,
                                                                                alpha_dict=alpha_dict_i,
                                                                                device=device,
                                                                                compute_loss=False,
                                                                                use_perfect_depth=params.use_perfect_depth)
                        if use_perfect_depth_map:
                            depth_i = all_zbuf[2+i:3+i]
                            error_mask_i = mask_i

                    depth.append(depth_i)
                    mask.append(mask_i)
                    error_mask.append(error_mask_i)


                depth = torch.cat(depth, dim=0)
                mask = torch.cat(mask, dim=0)
                error_mask = torch.cat(error_mask, dim=0)
                
                # ----------Build supervision signal from the new depth maps----------------------------------------------------
                all_part_pc = []
                all_part_pc_features = []
                all_fov_proxy_points = torch.zeros(0, 3, device=device)
                general_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()
                all_fov_proxy_mask = []
                all_sgn_dists = []
                all_X_cam = []
                all_fov_camera = []
                
                for i in range(depth.shape[0]):
                    fov_frame = camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
                    all_X_cam.append(fov_frame.get_camera_center())
                    all_fov_camera.append(fov_frame)

                    # TO CHANGE: filter points based on SSIM value!
                    part_pc, part_pc_features = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                images=batch_dict['images'][i:i+1],
                                                                mask=(mask * error_mask)[i:i + 1].bool(),
                                                                fov_cameras=fov_frame,
                                                                gathering_factor=params.gathering_factor,
                                                                fov_range=params.sensor_range)

                    # Surface points to fill surface scene
                    all_part_pc.append(part_pc)
                    all_part_pc_features.append(part_pc_features)

                    # Get Proxy Points in current FoV
                    fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                                                                                fov_camera=fov_frame, fov_range=params.sensor_range)
                    all_fov_proxy_points = torch.vstack((all_fov_proxy_points, fov_proxy_points))
                    all_fov_proxy_mask.append(fov_proxy_mask)
                    general_fov_proxy_mask = general_fov_proxy_mask + fov_proxy_mask

                    # Computing signed distance of proxy points in fov
                    sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[i:i + 1],
                                                                        mask=mask[i:i + 1].bool(), fov_camera=fov_frame
                                                                        ).view(-1, 1)
                    all_sgn_dists.append(sgn_dists)

                    # Computing mask for proxy points close to the surface.
                    # We will use this for occupancy probability supervision.
                    # close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

                # ----------Update Scenes to finalize supervision signal and prepare next iteration-----------------------------

                # 1. Surface scene
                # Fill surface scene
                # We give a visibility=1 to points that were visible in frame t, and 0 to others
                complete_part_pc = torch.vstack(all_part_pc)
                complete_part_pc_features = torch.vstack(all_part_pc_features)
                # complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
                # complete_part_pc_features[:len(all_part_pc[0])] = 1.
                surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

                full_pc = torch.vstack((full_pc, complete_part_pc))
                full_pc_colors = torch.vstack((full_pc_colors, complete_part_pc_features))

                # Compute coverage gain for each new camera pose
                # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
                # supervision_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
                # for i in range(depth.shape[0]):
                #     supervision_coverage_gains[i, 0] = surface_scene.camera_coverage_gain(all_part_pc[i],
                #                                                                           surface_epsilon=None)

                # # Update visibility history of surface points
                # surface_scene.set_all_features_to_value(value=1.)

                # 2. Proxy scene
                # Fill proxy scene
                general_fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
                proxy_scene.fill_cells(proxy_scene.proxy_points[general_fov_proxy_mask],
                                    features=general_fov_proxy_indices.view(-1, 1))

                for i in range(depth.shape[0]):
                    # Updating view_state vectors
                    proxy_scene.update_proxy_view_states(camera, all_fov_proxy_mask[i],
                                                        signed_distances=all_sgn_dists[i],
                                                        distance_to_surface=None,
                                                        X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

                    # Update the supervision occupancy for proxy points using the signed distance
                    proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i],
                                                            tol=params.carving_tolerance)

                # Update the out-of-field status for proxy points inside camera field of view
                proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)
                path_record += 1
                
                if pose_i == params.n_poses_in_trajectory:
                    #------plot occupancy map------
                    camera_pose_draw = torch.tensor([1, 0, 50, 30, 0]).to(device)
                    points_2d_batch_img = batch_transform_points_n_pieces(occupancy_X, camera_pose_draw, device)
                    current_camera_grids_batch_img = map_points_to_grid_optimized(points_2d_batch_img, occupancy_sigma, grid_size, grid_range)
                    
                    flipped_after_transpose = current_camera_grids_batch_img.squeeze(0)
                    
                    #------plot trajectory------
                    idx = len(camera.X_cam_history)-1
                    plt.figure(figsize=(6, 6), dpi=100)
                    # ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # plt.gcf().add_axes(ax)
                    ax = plt.gca()

                    x = gt_scene_pc[:, 0].cpu().numpy()
                    z = gt_scene_pc[:, 2].cpu().numpy()
                    x_min, x_max = np.min(x), np.max(x)
                    z_min, z_max = np.min(z), np.max(z)
                    buffer = 0
                    xlim = (x_min - buffer, x_max + buffer)
                    ylim = (z_min - buffer, z_max + buffer)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    # ax.set_xlabel('X Axis')
                    # ax.set_ylabel('Z Axis')
                    ax.scatter(x, z, s=1, color='black')
                    ax.invert_xaxis()

                    combined_camera_pose = torch.cat((camera.X_cam_history, camera.V_cam_history), dim=1)
                    combined_camera_pose = combined_camera_pose.cpu().numpy()

                    combined_camera_pose = combined_camera_pose[:idx+1]

                    previous_x, previous_z = None, None  

                    for i, pose in enumerate(combined_camera_pose):
                        pose[4:] = -np.deg2rad(pose[4:]-90)
                        x, _, z, _, azimuth = pose

                        if previous_x is not None and previous_z is not None:
                            ax.annotate('', xy=(x, z), xytext=(previous_x, previous_z),
                                arrowprops=dict(arrowstyle="->,head_length=1.2,head_width=0.8", linestyle="--", color="black"))
                        
                        previous_x, previous_z = x, z 
                        
                        color = 'black' if i == len(combined_camera_pose) - 1 else 'gray'
                        plt.scatter(x, z, color=color, s=60, edgecolors='none')
                        
                        line_length = 8
                        triangle_points = [[x, z]]
                        for angle in [np.pi / 7, -np.pi / 7]:
                            end_x = x + np.cos(azimuth+angle) * line_length
                            end_z = z + np.sin(azimuth+angle) * line_length
                            triangle_points.append([end_x, end_z])
                        
                        if color == 'black': 
                            ax.plot([x, triangle_points[1][0]], [z, triangle_points[1][1]], color='black', linewidth=0.5)
                            ax.plot([x, triangle_points[2][0]], [z, triangle_points[2][1]], color='black', linewidth=0.5)
                            
                            # 绘制黄色填充三角形，不包括边缘
                            triangle = patches.Polygon(triangle_points, closed=True, color='yellow', fill=True, edgecolor='none')
                        else:  # 其余位置使用灰色不填充的三角形，并尝试绘制虚线边缘
                            triangle = patches.Polygon(triangle_points, closed=True, fill=False, edgecolor='gray', linestyle='--')
                        ax.add_patch(triangle)
                    ax.axis('off')
                    plt.savefig(f'{folder_img_path}trajectory_{current_epoch}_{tra_num}.png') 
                    plt.clf()
                    #------plot heatmap---------    
                    gain_map_prediction = navi(flipped_after_transpose.unsqueeze(0).unsqueeze(0))
                    channels_to_plot = [
                        gain_map_prediction[:, 2, :, :],  # 第3个通道
                        gain_map_prediction[:, 1, :, :],  # 第2个通道
                        gain_map_prediction[:, 0, :, :],  # 第1个通道
                        gain_map_prediction[:, 3, :, :],  # 第4个通道
                        None,                             # 空白
                        gain_map_prediction[:, 0, :, :],  # 第0个通道（重复以匹配你的描述，可能需要调整）
                        gain_map_prediction[:, 5, :, :],  # 第5个通道
                        gain_map_prediction[:, 6, :, :],  # 第6个通道
                        gain_map_prediction[:, 7, :, :]   # 第7个通道
                    ]
                    # 计算全局最小值和最大值（除了中心图）
                    vmin = np.min([gain_map_prediction[:, i, :, :].cpu().min().item() for i in [0, 1, 2, 3, 5, 6, 7]])
                    vmax = np.max([gain_map_prediction[:, i, :, :].cpu().max().item() for i in [0, 1, 2, 3, 5, 6, 7]])

                    fig, axs = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})

                    for i, ax in enumerate(axs.flat):
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_aspect('auto')
                        if channels_to_plot[i] is None:
                            # 绘制中心图像
                            channel_image = flipped_after_transpose.cpu().numpy()
                            im = ax.imshow(channel_image, cmap='magma')  # 中心图使用不同的颜色映射
                        else:
                            # 绘制其他图像，使用全局的vmin和vmax
                            channel_image = channels_to_plot[i].squeeze().detach().cpu().numpy()
                            im = ax.imshow(channel_image, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
                        ax.axis('off')

                    # 添加一个统一的颜色条
                    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
                    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                    plt.savefig(f'{folder_img_path}mean_heatmap_{current_epoch}_{tra_num}.png') 
                    plt.clf()

                    #------plot results---------
                    
                    x = -full_pc[:, 0].cpu().numpy()
                    z = full_pc[:, 2].cpu().numpy()
                    plt.figure(figsize=(6, 6))
                    # 绘制点云的XZ平面视图
                    plt.scatter(x, z, s=1, color='black')
                    
                    plt.axis('off')
                    plt.gca().set_position([0, 0, 1, 1])
                    plt.gca().set_aspect('equal', adjustable='box')
                    
                    # 保存图像
            
                    plt.savefig(f'{folder_img_path}results_{current_epoch}_{tra_num}.png') 
                    plt.clf()  # 清除当前图形

