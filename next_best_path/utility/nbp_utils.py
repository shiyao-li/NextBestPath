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
from next_best_path.utility.nbp_utils import *
from next_best_path.utility.utils import *
from macarons.utility.macarons_utils import *
from macarons.trainers.train_macarons import setup_scene, setup_camera
from macarons.testers.scene import setup_test_scene, setup_training_camera
from next_best_path.utility.long_term_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms

import lmdb
import msgpack
import msgpack_numpy as m

def store_experience(env, data):
    with env.begin(write=True) as txn:
        key = f"{int(time.time()*1000):012d}".encode() 
        packed_data = msgpack.packb({
            'current_model_input': data['current_model_input'].cpu().numpy(),
            'current_gt_2d_layout': data['current_gt_2d_layout'].cpu().numpy(),
            'target_value_map_pixel': data['target_value_map_pixel'].cpu().numpy(),
            'actual_coverage_gain': data['actual_coverage_gain'].cpu().numpy(),
            'pose_i': np.array(data['pose_i'])
        }, use_bin_type=True)
        txn.put(key, packed_data)

def store_validation_data_readonly(env, num=600*2):
    selected_data = []
    with env.begin(write=False) as txn: 
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)  
        print("Number of total data in the database:", total_entries)
        
        n = math.ceil(total_entries / num)  
        
        count = 0
        for i, (key, value) in enumerate(cursor):
            if count % n == 0 and len(selected_data) < num:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))
                if len(selected_data) == num:  
                    break
            count += 1

    return selected_data

def read_random_data_readonly(env, num_samples=64):
    data = []
    indices = set(random.sample(range(env.stat()['entries']), num_samples)) 
    
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
    with env.begin(write=True) as txn: 
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)  
        print("Number of total data in the database:", total_entries)
        n = math.ceil(total_entries / num)  
        
        count = 0
        for i, (key, value) in enumerate(cursor):
            if count % n == 0 and len(selected_data) < num:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))
                delete_keys.append(key)
                if len(selected_data) == num:  
                    break
            count += 1

        for key in delete_keys:
            txn.delete(key)

    return selected_data


def read_combined_data(env, sample_m=2304*2):
    selected_data = []
    last_data = []
    total_entries = 0
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        
        for key, value in cursor:
            total_entries += 1
        print("number of total data in the database:", total_entries)
        
        if sample_m is None:
            cursor.first()
            return [msgpack.unpackb(value, object_hook=m.decode) for key, value in cursor]
        
        n = total_entries - sample_m

        if n < 0:
            n = 1
        
        cursor.first()
        sample_size = 2176*2
        sample_indices = random.sample(range(n), min(sample_size, n))  
        count = 0
        for i, (key, value) in enumerate(cursor):
            if i in sample_indices:
                selected_data.append(msgpack.unpackb(value, object_hook=m.decode))
            count += 1
            if count >= n:
                break
        
        if cursor.last():
            last_data.append(msgpack.unpackb(cursor.value(), object_hook=m.decode))
            for _ in range(sample_m - 1):
                if cursor.prev():
                    last_data.append(msgpack.unpackb(cursor.value(), object_hook=m.decode))
            last_data.reverse() 

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
        pred_gain_scaled = self.scaling_factor * pred_gain
        actual_gain_scaled = self.scaling_factor * actual_gain

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
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') #将权重初始化为正态分布，均值为0，方差为sqrt（2/输入单元数），非线性激活函数为relu
        # m.bias.data.fill_(0.01)
        # torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        # torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.weight, 0., 0.005) #0.005)
        # m.bias.data.fill(0.01)

def initialize_nbp(params, nbp, torch_seed, initialize, 
                     pretrained, ddp_rank):
    model_name = params.nbp_model_name
    start_epoch = 0
    best_loss = 10000.
    
    # Weight initialization if needed
    if initialize:

        nbp.apply(init_weights)
                
    else: 
        pass

    # 收缩前：params = params - lr*weight_decay*params 
    # 收缩后：params = params - lr*grad(params)
    # Delta w = lr * m_t / (sqrt(v_t) + epsilon)
    # m_t = beta1(0.9) * m_{t-1} + (1 - beta1) * grad(params)
    # v_t = beta2(0.999) * v_{t-1} + (1 - beta2) * grad(params)^2
    optimizer = optim.AdamW(nbp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    

    return nbp, optimizer, best_loss, start_epoch

def setup_macarons(params, macarons_model_path, device):
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_macarons_path, device=device, learn_pose=params.learn_pose)
    trained_weights = torch.load(macarons_model_path, map_location=device)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True) 
    return macarons

def setup_memory(params, train_set, train_dataloader):
    # 创建记忆
    print("\n使用记忆文件夹: ", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in train_set: 
        scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path) # 添加记忆路径 例如：training_dataset\appalling_icons_0\macarons_memory
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories, # 轨迹数量
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
    
    full_transform = transforms.RandomApply([transform], p=0.4)

    seed = torch.random.initial_seed()
    torch.manual_seed(seed)
    augmented_gt_layout = full_transform(gt_layout)

    torch.manual_seed(seed)
    augmented_input = full_transform(input_tensor)

    return augmented_gt_layout, augmented_input



def validation_model(training_set_db, params, nbp, device):
    criterion_maps = nn.MSELoss()
    criterion_layout = nn.BCELoss()

    accumulated_loss = 0
    update_count = 0

    for i in range(0, len(training_set_db), params.nbp_batch_size):
        batch_data = training_set_db[i:i + params.nbp_batch_size]
        training_exp = []
        for data in batch_data:
            current_model_input = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
            current_gt_2d_layout = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)
            target_heatmap_pixel = torch.from_numpy(np.copy(data['target_value_map_pixel'])).to(device)
            actual_path_gain = torch.from_numpy(np.copy(data['actual_coverage_gain'])).to(device)
            training_exp.append([
                current_model_input,
                current_gt_2d_layout,
                target_heatmap_pixel,
                actual_path_gain
            ])

        input_images = torch.cat([data[0] for data in training_exp])
        layouts = torch.cat([data[1] for data in training_exp])
        coords_list = [data[2] for data in training_exp]
        gt_pixels_list = [data[3] for data in training_exp]

        batch_coords = torch.cat(coords_list).to(device)
        batch_gt_pixels = torch.cat(gt_pixels_list).to(device)
        batch_sizes = [len(coords) for coords in coords_list]
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), torch.tensor(batch_sizes, device=device))

        pred_heatmaps, pred_layouts = nbp(input_images)
        pred_values = pred_heatmaps[batch_indices, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]]

        heatmaps_loss = criterion_maps(pred_values, batch_gt_pixels)
        layout_loss = criterion_layout(pred_layouts, layouts)
        batch_loss = heatmaps_loss + layout_loss

        accumulated_loss += batch_loss.item()
        update_count += 1

    validation_loss = accumulated_loss / update_count
    print("Validation loss:", validation_loss)

    return validation_loss

def train_experience_data(training_set_db, params, optimizer, nbp, device, current_epoch):
    random.shuffle(training_set_db)
    scaler = GradScaler()
    training_loss = []
    accumulated_loss = 0
    accumulation_steps = 8 # gaile: 8
    update_count = 0

    for i in range(0, len(training_set_db), params.nbp_batch_size):
        batch_data = training_set_db[i:i + params.nbp_batch_size]
        training_exp = []
        for data in batch_data:
            if (data['pose_i'] > 10 and current_epoch == 1) or current_epoch > 1:  # current_epoch > 1
                current_model_input = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
                current_gt_obs_layout = torch.from_numpy(np.copy(data['current_gt_2d_layout'])).to(device)
                target_value_map_pixel = torch.from_numpy(np.copy(data['target_value_map_pixel'])).to(device)
                actual_coverage_gain = torch.from_numpy(np.copy(data['actual_coverage_gain'])).to(device)

                training_exp.append([
                    current_model_input,
                    current_gt_obs_layout,
                    target_value_map_pixel,
                    actual_coverage_gain
                ])
        
        if not training_exp:
            continue

        input_images = torch.cat([data[0] for data in training_exp])
        gt_obs = torch.cat([data[1] for data in training_exp])
        coords_list = [data[2] for data in training_exp]
        gt_pixels_list = [data[3] for data in training_exp]

        batch_coords = torch.cat(coords_list).to(device)
        batch_gt_pixels = torch.cat(gt_pixels_list).to(device)
        batch_sizes = [len(coords) for coords in coords_list]
        batch_indices = torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), torch.tensor(batch_sizes, device=device))

        predicted_value_map, predicted_obs_map = nbp(input_images)
        pred_values = predicted_value_map[batch_indices, batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]]

        batch_loss = nbp.loss(pred_values, batch_gt_pixels, predicted_obs_map, gt_obs)

        scaler.scale(batch_loss).backward()
        accumulated_loss += batch_loss.item()
        update_count += 1

        if update_count % accumulation_steps == 0 or (i + params.nbp_batch_size) >= len(training_set_db):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            training_loss.append(accumulated_loss / accumulation_steps)
            accumulated_loss = 0
            update_count = 0

    return training_loss

def evaluate_experience_data(all_experiences, params, nbp, average_kernel, kernel_size, device):
    validation_losses = []
    total_batches = len(all_experiences) // params.nbp_batch_size + (1 if len(all_experiences) % params.nbp_batch_size > 0 else 0)
    criterion_maps = nn.MSELoss() 
    criterion_layout = nn.BCELoss()  
    random.shuffle(all_experiences)
    for batch_index in range(total_batches):
        start_index = batch_index * params.nbp_batch_size
        end_index = start_index + params.nbp_batch_size
        experiences = all_experiences[start_index:end_index]

        batch_start_grids, batch_previous_trajectories, batch_gt_2d_layouts, batch_target_locations, batch_angles, batch_gains = zip(*experiences)
        batch_start_grids = torch.cat(batch_start_grids).to(device)
        batch_previous_trajectories = torch.cat(batch_previous_trajectories).to(device)
        batch_gt_2d_layouts = torch.cat(batch_gt_2d_layouts).to(device)
        batch_target_locations = torch.cat(batch_target_locations).to(device)
        batch_angles = torch.cat(batch_angles).to(device)
        batch_gains = torch.tensor(batch_gains, dtype=torch.float32, device=device)

        output_maps = nbp(torch.cat((batch_start_grids, batch_previous_trajectories), dim=1).to(device))

        # batch_loss = torch.mean(torch.stack(mse_losses))
        indices = torch.arange(len(batch_angles), device=device)
        predicted_potentials = output_maps[indices, batch_angles, batch_target_locations[:, 0], batch_target_locations[:, 1]]
        loss_maps = criterion_maps(predicted_potentials, batch_gains)
        # loss_layout = criterion_layout(output_layout, batch_gt_2d_layouts) 
        batch_loss = loss_maps # + loss_layout 
        validation_losses.append(batch_loss.item())

    return np.mean(validation_losses)


#---------Functons to train the model------------
def train_nbp(db_env, params, optimizer, nbp, device, folder_img_path, current_epoch, validation_data, last_experience_num=2304, validation_split=0.1, patience=10, lr_patience=2, lr_factor=0.1):
    
    train_losses = []
    validation_losses = []


    if True: 
        if current_epoch == 1:
            last_experience_num = None
            training_set_db = read_combined_data(db_env, sample_m=last_experience_num)
        else:
            training_set_db = read_combined_data(db_env)
            print(len(training_set_db))


        num_epochs = 5
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True)

        for epoch in range(num_epochs):
            nbp.train()
            
            train_loss = train_experience_data(training_set_db=training_set_db, params=params, optimizer=optimizer, nbp=nbp, device=device, current_epoch=current_epoch)
            nbp.eval()
            with torch.no_grad():
                validation_loss = validation_model(validation_data, params, nbp, device)

            print(np.mean(train_loss))
            train_losses.append(np.mean(train_loss))
            validation_losses.append((validation_loss))

            lr_scheduler.step((validation_loss))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}")

    average_loss = sum(train_losses) / len(train_losses)
    average_validation_loss = sum(validation_losses) / len(validation_losses)
    del training_set_db
    gc.collect()
    return average_loss, average_validation_loss

def trajectory_collection(params, current_epoch, train_dataloader, db_env, pc2img_size, value_map_size, prediction_range,
                        nbp, coverage_after_trajectory, memory, device, folder_img_path):
    
    use_perfect_depth_map = True # 使用完美深度图，避免深度预测误差
    num_scenes = 0 # 场景计数
    for batch, scene_dict in enumerate(train_dataloader):
        if (num_scenes+1) % 35 == 0:
            print(num_scenes)
        # 场景计数加1
        num_scenes += 1
        
        scene_names = scene_dict['scene_name']
        obj_names = scene_dict['obj_name']
        all_settings = scene_dict['settings']
        # occupied_pose_datas = scene_dict['occupied_pose']
        batch_size = 1 # 对一个场景做一次轨迹采集
        
        for i_scene in range(batch_size):
            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            # occupied_pose_data = occupied_pose_datas[i_scene]
            # print("\nScene name:", scene_name)
            # print("-------------------------------------")
            
            scene_path = os.path.join(train_dataloader.dataset.data_path, scene_name) # 获取场景路径
            mesh_path = os.path.join(scene_path, obj_name) # 获取模型路径
            
            #---ToChange: mirror scenes
            mirrored_scene = False # 按照某条轴（x,y,z）做镜像（bool），If True, mirrors the coordinates on x-axis
            mirrored_axis = None # 按照某条轴（x,y,z）做镜像
            
            # 把当前3D场景加载成带纹理的PyTorch Meshes对象，供后续渲染和轨迹使用。
            mesh = load_scene_with_texture(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)

            # 将点云分割成n个部分，沿着y轴分割
            n_pieces = 4
            verts = mesh.verts_list()[0] # 获取当前mesh的顶点（N，3），N为顶点数
            # 获取当前mesh的顶点沿着y轴的最小值和最大值（0.5是为了避免顶点重合）
            min_y, max_y = torch.min(verts, dim=0)[0][1].item() + 0.5, torch.max(verts, dim=0)[0][1].item() - 0.5

            bin_width = (max_y - min_y) / n_pieces # 计算每个分割的宽度
            y_bins = torch.arange(min_y, max_y+bin_width, bin_width, device=device) # 计算每个分割的边界,4段5个分界点


            # 存储一个mesh用于碰撞检查，例如相机位置是否在mesh内，两点之间是否穿墙，Dijkstra路径是否合法
            mesh_for_check = trimesh.load(mesh_path) 
            mesh_for_check.vertices *= params.scene_scale_factor
            
            # 使用内存信息设置帧和位姿路径
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories # 获取当前epoch的轨迹编号
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb) # 获取当前轨迹的帧路径
            
            gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
            test_resolution = 0.05 # 体素/网格的分别率，影响表面网格的精度
            gc.collect() # 回收内存
            torch.cuda.empty_cache() # 清空CUDA缓存
            
            """
    场景对象	     职责描述	                数据来源	            特征维度 (feature_dim)	 特征含义
    gt_scene	    性能评估的客观基准	        导入的真实 3D Mesh	      3	                      RGB 真实颜色
    surface_scene	存储模型推断的              3D 表面	深度模型 + RGB 图像预测	1 (通常)	       可见性历史 (Visibility)
    covered_scene	记录场景的探索/覆盖状态	    动态更新的观测记录	        1 (通常)	            探索覆盖状态 (Coverage)
    proxy_scene	    渲染加速或路径规划的辅助点	 空间初始化逻辑生成	        1                       代理点索引 (Indices)
    """
            gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                    mesh,
                                                                                    settings,
                                                                                    mirrored_scene,
                                                                                    device,
                                                                                    mirrored_axis=None,
                                                                                    surface_scene_feature_dim=3,#表面场景的特征维度
                                                                                    test_resolution=test_resolution) #测试分辨率
            
            start_cam_idx = settings.camera.start_positions[0] # 获取起始相机索引
            
            # sstup_training_2d_camera
            occupied_pose_data = None
            # 设置训练相机
            camera = setup_training_camera(params, mesh, mesh_for_check, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
            

            full_pc = torch.zeros(0, 3, device=device) # 初始化全点云
            full_pc_colors = torch.zeros(0, 3, device=device) # 初始化全点云颜色

            coverage_evolution = [] # 覆盖率演变
            

            Dijkstra_path = [] # Dijkstra路径
            path_record = 0 # 路径记录
            unreachable_position = [] # 不可达位置
            
            # Split camera dictionary to: {key, (first_tensor, second_tensor)}
            '''
            splited_pose_space_idx = {
    '[8, 0, 8, 0]': (tensor_A, tensor_B),   # 位置+朝向的某种表示
    '[8, 0, 8, 1]': (tensor_A, tensor_B),
    '[7, 0, 9, 2]': (...),
    # ... 很多个合法位姿
}
            '''
            splited_pose_space_idx = camera.generate_new_splited_dict()
            
            # splited_pose_space: '[8,  0,  8]': tensor([43.5556,  0.0000, 93.5556], device='cuda:1')
            splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

            # compute the bounding box for the gt_pc, (20000, 3)
            gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)
            
            experiences_list = [] # 经验列表
            # 遍历100个位姿，单条轨迹的探索主循环
            for pose_i in range(100):
                # 加载图像和深度图
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                                n_frames=1,
                                                                                                n_alpha=params.n_alpha,
                                                                                                return_gt_zbuf=True)                          
                                                                                                
                current_coverage = calculate_coverage_percentage(gt_scene_pc, full_pc) # 计算覆盖率
                coverage_evolution.append(current_coverage) # 记录覆盖率
                    
                if pose_i == params.n_poses_in_trajectory: # 如果到达轨迹终点
                    coverage_after_trajectory.append(current_coverage) # 记录覆盖率

                if current_coverage > 0.95: # 如果覆盖率大于0.95，则结束轨迹
                    break
                        
                torch.cuda.empty_cache() # 清空CUDA缓存
                # 创建深度模型输入
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                                all_images=all_images, all_mask=all_mask,
                                                                all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                                mode='inference', device=device,
                                                                all_zbuf=all_zbuf)

                # 深度预测
                with torch.no_grad():
                    depth, mask, error_mask, pose, gt_pose = obtain_depth(params=params,
                                                                batch_dict=batch_dict,
                                                                alpha_dict=alpha_dict,
                                                                device=device,
                                                                use_perfect_depth=params.use_perfect_depth)
                    
                # 如果使用完美深度图，则使用真实深度图代替预测深度图
                if use_perfect_depth_map:
                    depth = all_zbuf[2:3]
                    error_mask = mask

                # We fill the surface scene with the partial point cloud
                for i in range(depth.shape[0]): #这里只有一帧，因为params.n_interpolation_steps == 1
                    # TO CHANGE: filter points based on SSIM value!
                    # 把深度图反投影成 3D
                    part_pc, part_pc_features = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                                images=batch_dict["images"][i:i+1],
                                                                mask=(mask * error_mask)[i:i + 1],
                                                                fov_cameras=camera.get_fov_camera_from_RT(
                                                                    R_cam=batch_dict['R'][i:i + 1],
                                                                    T_cam=batch_dict['T'][i:i + 1]),
                                                                gathering_factor=params.gathering_factor,
                                                                fov_range=params.sensor_range)

                    full_pc = torch.vstack((full_pc, part_pc)) # 将当前帧的点云添加到全点云中
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_features)) # 将当前帧的点云颜色添加到全点云颜色中

                camera_current_pose, _ = camera.get_pose_from_idx(camera.cam_idx) # 获取当前相机位姿

                #---------------------------------------------------------------------------------------------------------
                # NBP model input preparation: projected images of current point cloud and previous trajectory
                #---------------------------------------------------------------------------------------------------------
                # We first divide current full_pc into n pieces along Y-axis

                # 将全点云分割成n个部分，沿着y轴分割，bins，形状 (5000,)，取值 0,1,2,3。
                bins = torch.bucketize(full_pc[:, 1], y_bins[:-1]) - 1 
                # 长度为4的数组，每个元素为full_pc的一部分，形状 (N, 3)，N为点云数
                full_pc_groups = [full_pc[bins == i] for i in range(n_pieces)]
                full_pc_images = []

                # 将3D图映射成2D图，再把2D图按照prediction_range映射到pc2img_size的范围内，得到4个2D图，形状 (4, 100, 100)
                for i in range(n_pieces):
                    if len(full_pc_groups[i]) > 0:
                        points_2d_batch = transform_points_to_n_pieces(full_pc_groups[i], camera_current_pose, device)
                        current_partial_pc_img = map_points_to_n_imgs(points_2d_batch, pc2img_size, prediction_range, device) # size(1, 100, 100)
                    else:
                        current_partial_pc_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device)
                    full_pc_images.append(current_partial_pc_img)

                # 将4个2D图拼接成1个2D图，形状 (4, 100, 100)
                full_pc_images = torch.cat(full_pc_images, dim=0)
                current_pc_imgs = full_pc_images.unsqueeze(0) # 形状 (1, 4, 100, 100)


                # gt obstacle map collection
                # 获取障碍物地图，形状 (1, 100, 100)
                obs_array = get_binary_obstacle_array(mesh_for_check, camera_current_pose, view_size=prediction_range[1]*2)
                current_gt_obs = torch.tensor(obs_array, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # previsous camera trajectory collection:
                # 将历史轨迹映射成2D图，形状 (N, 2)，N为轨迹点数
                trajectory_2d = transform_points_to_n_pieces(camera.X_cam_history, camera_current_pose, device)
                # 将2D轨迹映射成2D图，形状 (1, 100, 100)
                previous_trajectory_img = map_points_to_n_imgs(trajectory_2d, pc2img_size, prediction_range, device)
                # 形状 (1, 1, 100, 100)
                current_previous_trajectory_img = previous_trajectory_img.unsqueeze(0)

                ############################################################################################################
                # Dijkstra path generation and training data collection
                ############################################################################################################
           
                # 写 LMDB：上一段路径 → 训练样本
                if path_record + 1 > len(Dijkstra_path):
                    
                    # nbp模型数据收集，完成最后一条轨迹后，收集训练样本
                    if len(experiences_list) > 0:
                        for ex in range(len(experiences_list)):
                            # 在 ex 时刻的输入下，NBP 在 ex_next 相机位置对应的像素上，应该预测出从 ex 到 ex_next 的真实覆盖率增量
                            if ex+1 <= len(experiences_list):
                                pixel_list = [] # 在最后一条轨迹中，可以得到的真实像素值列表
                                value_list = []

                                for ex_next in range(ex+1, len(experiences_list)):
                                    # 用 experiences_list[ex][-2]（ex 时刻的相机位姿）做参考，把该 3D 点变换到 ex 时刻的视角下的 2D 位置。
                                    ex_next_location = transform_points_to_n_pieces(experiences_list[ex_next][-2][:3].unsqueeze(0), experiences_list[ex][-2], device)
                                    # 把上面 2D 位置映射到 value map 的网格坐标 (iy, ix)，范围由 value_map_size、prediction_range 决定。
                                    ex_grid_position = get_point_position_in_the_img(ex_next_location.squeeze(0), value_map_size, prediction_range)

                                    # 若 (iy, ix) 在 [0, value_map_size[0]) 和 [0, value_map_size[1]) 内，才当作有效样本。
                                    if 0 <= ex_grid_position[0] < value_map_size[0] and 0 <= ex_grid_position[1] < value_map_size[1]:

                                        # 有效时，把 (iy, ix) 做成 tensor，即 value map 上的像素坐标。
                                        ex_cam_img_position = torch.tensor([ex_grid_position[0], ex_grid_position[1]]).to(device)

                                        # 从 ex 走到 ex_next 的真实覆盖率增量： (experiences_list[ex_next][0] - experiences_list[ex][0]) * 100，且若 ≤0 则取 0。
                                        actual_coverage_gain = (experiences_list[ex_next][0]-experiences_list[ex][0])*100 if (experiences_list[ex_next][0]-experiences_list[ex][0]) > 0 else 0
                                        # 把“通道 + 行 + 列”拼成一个向量，训练时组 batch 得到 batch_coords，形状 (N, 3)
                                        current_pixel = torch.cat((experiences_list[ex_next][-1].unsqueeze(0), ex_cam_img_position), dim=0) # 把 ex_next 时刻的相机位姿和 (iy, ix) 拼接成一个 tensor，形状 (2,)
                                        pixel_list.append(current_pixel)
                                        value_list.append(actual_coverage_gain)
                                if len(pixel_list) > 0: 
                                    pixels = torch.stack(pixel_list, dim=0) # 把 pixel_list 中的所有 tensor 拼接成一个 tensor，形状 (N, 3)
                                    stack_gains = torch.tensor(value_list, dtype=torch.float32, device=device) # 把 value_list 中的所有 float 拼接成一个 tensor，形状 (N,)
                                    experience_db = {
                                        'current_model_input': experiences_list[ex][1],
                                        'current_gt_2d_layout': experiences_list[ex][2],
                                        'target_value_map_pixel': pixels,
                                        'actual_coverage_gain': stack_gains,
                                        'pose_i' : pose_i
                                    }
                                    store_experience(db_env, experience_db) # 把经验写入 LMDB


                                    
                        experiences_list = []    
                        
                    Dijkstra_path = []
                    path_record = 0
                    
                    # NBP 前向 + 选目标位姿
                    # 用当前的 NBP 输出 value map，对所有候选相机位置评估“价值”，用 Boltzmann 采样选一个目标位置放到列表最前面，供后面 Dijkstra 生成新路径。


                    current_model_input = torch.cat((current_pc_imgs, current_previous_trajectory_img), dim=1).to(device)
                    predicted_value_map, _ = nbp(current_model_input)

                    max_gain_map, _ = torch.max(predicted_value_map, dim=1, keepdim=True)
                    camera_position_value_list = []

                    # 遍历所有可能的相机位置，计算每个位置的最佳价值
                    for key, point_3d in splited_pose_space.items():
                        if camera.cam_idx[:3].tolist() != ast.literal_eval(key):

                            # point_2d ≈ tensor([[[ 5.0, 10.0]]])   # 某个相机坐标系下的 2D 点
                            point_2d = transform_points_to_n_pieces(point_3d.unsqueeze(0), camera_current_pose, device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')

                            # 假设映射到 value map 网格是 (iy=32, ix=40)
                            # value_map_size = (64, 64)
                            grid_position = get_point_position_in_the_img(point_2d.squeeze(0), value_map_size, prediction_range)
                            
                            """
                            cam_img_position = tensor([32, 40])   # (y, x)
                            average_cam_img_result = max_gain_map[0, 0, 32, 40].detach()
                            # 假设 = tensor(0.5)

                            new_position_values = ['[7,0,9]', tensor([32,40]), tensor(0.5)]
                            camera_position_value_list.append(new_position_values)
                            """
                            if 0 <= grid_position[0] < value_map_size[0] and 0 <= grid_position[1] < value_map_size[1]:
                                # Change to locations in image: (H - 1 - y, x)
                                cam_img_position = torch.tensor([grid_position[0], grid_position[1]]).to(device)

                                average_cam_img_result = max_gain_map[0, 0, cam_img_position[0], cam_img_position[1]].detach()
                                # Store these info to a list
                                new_position_values = []
                                new_position_values.append(key)
                                new_position_values.append(cam_img_position)
                                new_position_values.append(average_cam_img_result)
                                camera_position_value_list.append(new_position_values)
                            
                    # Boltzmann exploration for training
                    # gain_values = tensor([0.5, 0.3])   # shape (2,)
                    gain_values = torch.stack([item[2] for item in camera_position_value_list])
                    beta = 0.5  # temperature parameter


                    # gain_values / beta = [1.0, 0.6]
                    # exp 后大致是 [e^1.0, e^0.6] ≈ [2.718, 1.822]
                    # 归一化:
                    # probabilities ≈ [2.718/(2.718+1.822), 1.822/(2.718+1.822)]
                    #             ≈ [0.598, 0.402]
                    probabilities = torch.softmax(gain_values / beta, dim=0)
                    selected_idx = torch.multinomial(probabilities, 1).item()
                    selected_item = camera_position_value_list.pop(selected_idx)
                    camera_position_value_list.insert(0, selected_item)
                    

                    # 生成 Dijkstra 路径并更新 experiences_list
                    # path_start_position = [8, 0, 8]   # 当前相机所在格子的索引
                    path_start_position = camera.cam_idx[:3].tolist()
                    for pose_location in camera_position_value_list:
                        # pose_location = ['[6,1,8]', tensor([10,50]), tensor(0.3)]
                        # path_end_position = [6, 1, 8]
                        path_end_position = ast.literal_eval(pose_location[0])
                        # 检查目标点是否在 mesh 内
                        if check_camera_in_mesh(mesh_for_check, splited_pose_space[pose_location[0]]):
                            # 检查目标点是否在不可达位置列表中
                            if path_end_position in unreachable_position:
                                continue
                            # 考虑 mesh 障碍和 value map，在离散格点上做最短路径
                            Dijkstra_path = generate_Dijkstra_path(splited_pose_space, path_start_position, path_end_position, mesh_for_check, camera_current_pose, camera,
                                                                    value_map_size, prediction_range, predicted_value_map, device)
                            """
                            Dijkstra_path = [
                                [8,0,8],   # 当前（或下一步）
                                [7,0,8],
                                [7,1,8],
                                [6,1,8],   # 目标
                            ]
                            """
                            if Dijkstra_path is not None:
                                experiences_list.append([
                                    coverage_evolution[-1],   # 当前覆盖率，如 0.35
                                    current_model_input,      # (1,5,H,W)
                                    current_gt_obs,          # (1,1,H,W)
                                    camera_current_pose,      # 4×4 或 R,T
                                    camera.cam_idx[-1]        # 朝向索引，如 3
                                ])
                                break
                            else:
                                # unreachable_position = [[7,0,9], [6,1,8]]
                                # 记录下无法到达的路径，避免重复尝试
                                unreachable_position.append(path_end_position)
                            
                else:
                    experiences_list.append([
                                        coverage_evolution[-1],
                                        current_model_input,
                                        current_gt_obs,
                                        camera_current_pose,
                                        camera.cam_idx[-1]
                                    ])
                """
                Dijkstra_path = [
                    [8,0,8,2],   # 索引 0
                    [7,0,8,1],   # 索引 1
                    [7,1,8,0],   # 索引 2
                    [6,1,8,3],   # 索引 3
                ]
                path_record = 1（上一步已经走到索引 1，本步要走到索引 2）。
                full_pc 当前形状 (12000, 3)，full_pc_colors (12000, 3)。
                n_interpolation_steps = 4，use_perfect_depth_map = True。
                """
                # 若本步没有可行路径（前面所有候选都失败），直接结束本轨迹的 pose_i 循环。
                if Dijkstra_path is None:
                    break
                

                if path_record >= len(Dijkstra_path):
                    print(f"IndexError: Index {path_record} out of range for Dijkstra_path with length {len(Dijkstra_path)}.")
                    break

                # 本步要去的位姿索引，例如 [7, 0, 8, 3]（前 3 个是格子，最后 1 个是朝向）。
                # next_idx = Dijkstra_path[1]   # [7, 0, 8, 1]
                next_idx = Dijkstra_path[path_record]

                # 以 60% 概率把朝向随机成 0～7 之一，做探索/数据增强。
                if random.random() <= 0.6:
                    next_idx[-1] = torch.randint(0, 8, (1,))

                # 相机从“上一步位置”平滑移到 next_idx，并得到多帧 RGB/深度等（用于后面算这一段路径上的点云）。
                # 相机从当前位姿逐步插值到 [7,0,8,1]，并在 4 个中间/终点位姿渲染，得到 4 帧的 RGB 和深度等。
                interpolation_step = 1
                for i in range(camera.n_interpolation_steps):
                    camera.update_camera(next_idx, interpolation_step=interpolation_step)
                    camera.capture_image(mesh)
                    interpolation_step += 1
                    

                # 处理机器人在刚刚那段短途移动中（沿途）拍下的多张照片，并把每一张照片都转换成深度图（Depth Map）。
                # 对刚移动到的这一段轨迹上的多帧，加载 RGB、zbuf、mask、R、T、zfar
                # all_images: (4, 3, 256, 456) 等
                # all_zbuf: (4, 1, 256, 456) 等
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
                                camera=camera,
                                n_frames=params.n_interpolation_steps,
                                n_alpha=params.n_alpha_for_supervision,
                                return_gt_zbuf=True)
                # 把多帧打成 batch_dict、alpha_dict，供深度模块用。
                # batch_dict['images']: (4, 3, 256, 456)
                # batch_dict['R']: (4, 3, 3), batch_dict['T']: (4, 3) 等
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=device, all_zbuf=all_zbuf)
                # 用 batch_dict、alpha_dict 跑深度模型，得到多帧深度图 depth、mask、error_mask
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
                        depth_i, mask_i, error_mask_i, _, _ = obtain_depth(params=params,
                                                                batch_dict=batch_dict,
                                                                alpha_dict=alpha_dict,
                                                                device=device,
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
                

                # 逐帧反投影成点云并拼进 full_pc（911–933）
                all_part_pc = []
                all_part_pc_features = []
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

                    all_part_pc.append(part_pc)
                    all_part_pc_features.append(part_pc_features)
                complete_part_pc = torch.vstack(all_part_pc)
                complete_part_pc_features = torch.vstack(all_part_pc_features)

                full_pc = torch.vstack((full_pc, complete_part_pc))
                full_pc_colors = torch.vstack((full_pc_colors, complete_part_pc_features))

                path_record += 1


