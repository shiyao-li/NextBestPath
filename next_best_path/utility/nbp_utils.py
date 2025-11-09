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
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
        # torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        # torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.weight, 0., 0.005) #0.005)
        # m.bias.data.fill(0.01)

def initialize_nbp(params, nbp, torch_seed, initialize, 
                     pretrained, ddp_rank):
    model_name = params.navi_model_name
    start_epoch = 0
    best_loss = 10000.
    
    # Weight initialization if needed
    if initialize:

        nbp.apply(init_weights)
                
    else: 
        pass

    
    optimizer = optim.AdamW(nbp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    

    return nbp, optimizer, best_loss, start_epoch

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
            target_heatmap_pixel = torch.from_numpy(np.copy(data['target_heatmap_pixel'])).to(device)
            actual_path_gain = torch.from_numpy(np.copy(data['actual_path_gain'])).to(device)
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

    for i in range(0, len(training_set_db), params.navi_batch_size):
        batch_data = training_set_db[i:i + params.navi_batch_size]
        training_exp = []
        for data in batch_data:
            if (data['pose_i'] > 10 and current_epoch == 1) or current_epoch > 1:  # current_epoch > 1
                current_model_input = torch.from_numpy(np.copy(data['current_model_input'])).to(device)
                current_gt_obs_layout = torch.from_numpy(np.copy(data['current_gt_obs_layout'])).to(device)
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
    criterion_maps = nn.MSELoss() 
    criterion_layout = nn.BCELoss()  
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
            
            train_loss = train_experience_data(training_set_db=training_set_db, params=params, optimizer=optimizer, navi=nbp, device=device, current_epoch=current_epoch)
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
            camera = setup_training_camera(params, mesh, mesh_for_check, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
            

            full_pc = torch.zeros(0, 3, device=device)
            full_pc_colors = torch.zeros(0, 3, device=device)

            coverage_evolution = []
            

            Dijkstra_path = []
            path_record = 0
            unreachable_position = []
            
            # Split camera dictionary to: {key, (first_tensor, second_tensor)}
            splited_pose_space_idx = camera.generate_new_splited_dict()
            
            # splited_pose_space: '[8,  0,  8]': tensor([43.5556,  0.0000, 93.5556], device='cuda:1')
            splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)

            # compute the bounding box for the gt_pc
            gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)
            
            experiences_list = []
            
            for pose_i in range(100):
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                                n_frames=1,
                                                                                                n_alpha=params.n_alpha,
                                                                                                return_gt_zbuf=True)                          
                                                                                                
                current_coverage = calculate_coverage_percentage(gt_scene_pc, full_pc)
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
                    depth, mask, error_mask, pose, gt_pose = obtain_depth(params=params,
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

                    full_pc = torch.vstack((full_pc, part_pc))
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_features))

                camera_current_pose, _ = camera.get_pose_from_idx(camera.cam_idx)

                #---------------------------------------------------------------------------------------------------------
                # NBP model input preparation: projected images of current point cloud and previous trajectory
                #---------------------------------------------------------------------------------------------------------
                # We first divide current full_pc into n pieces along Y-axis
                bins = torch.bucketize(full_pc[:, 1], y_bins[:-1]) - 1 # compute the range of n_pieces: exact the index of y_values
                full_pc_groups = [full_pc[bins == i] for i in range(n_pieces)]
                full_pc_images = []

                for i in range(n_pieces):
                    if len(full_pc_groups[i]) > 0:
                        points_2d_batch = transform_points_to_n_pieces(full_pc_groups[i], camera_current_pose, device)
                        current_partial_pc_img = map_points_to_n_imgs(points_2d_batch, pc2img_size, prediction_range, device) # size(1, 100, 100)
                    else:
                        current_partial_pc_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device)
                    full_pc_images.append(current_partial_pc_img)

                full_pc_images = torch.cat(full_pc_images, dim=0)
                current_pc_imgs = full_pc_images.unsqueeze(0)

                # gt obstacle map collection
                obs_array = get_binary_obstacle_array(mesh_for_check, camera_current_pose, view_size=prediction_range[1]*2)
                current_gt_obs = torch.tensor(obs_array, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                # previsous camera trajectory collection:
                trajectory_2d = transform_points_to_n_pieces(camera.X_cam_history, camera_current_pose, device)
                previous_trajectory_img = map_points_to_n_imgs(trajectory_2d, pc2img_size, prediction_range, device)
                current_previous_trajectory_img = previous_trajectory_img.unsqueeze(0)

                ############################################################################################################
                # Dijkstra path generation and training data collection
                ############################################################################################################
           

                if path_record + 1 > len(Dijkstra_path):
                    
                    # Data collection for nbp model, we collect the training data after finishing the last trajectory:
                    if len(experiences_list) > 0:
                        for ex in range(len(experiences_list)):
                            # apply data augmentation: search a pixels in a single map
                            if ex+1 <= len(experiences_list):
                                pixel_list = [] # the gt pixels values in the value map we can get in the last trajectory
                                value_list = []

                                # here, we do the data agumentation: every subset of the dijkstra path is the dijkstra path
                                for ex_next in range(ex+1, len(experiences_list)):
                                    ex_next_location = transform_points_to_n_pieces(experiences_list[ex_next][-2][:3].unsqueeze(0), experiences_list[ex][-2], device)
                                    ex_grid_position = get_point_position_in_the_img(ex_next_location.squeeze(0), value_map_size, prediction_range)
                                    if 0 <= ex_grid_position[0] < value_map_size[0] and 0 <= ex_grid_position[1] < value_map_size[1]:
                                        ex_cam_img_position = torch.tensor([ex_grid_position[0], ex_grid_position[1]]).to(device)
                                        # Avoid actual coverage gain smaller than 0
                                        actual_coverage_gain = (experiences_list[ex_next][0]-experiences_list[ex][0])*100 if (experiences_list[ex_next][0]-experiences_list[ex][0]) > 0 else 0

                                        current_pixel = torch.cat((experiences_list[ex_next][-1].unsqueeze(0), ex_cam_img_position), dim=0)
                                        pixel_list.append(current_pixel)
                                        value_list.append(actual_coverage_gain)
                                if len(pixel_list) > 0: 
                                    pixels = torch.stack(pixel_list, dim=0)
                                    stack_gains = torch.tensor(value_list, dtype=torch.float32, device=device)
                                    experience_db = {
                                        'current_model_input': experiences_list[ex][1],
                                        'current_gt_2d_layout': experiences_list[ex][2],
                                        'target_value_map_pixel': pixels,
                                        'actual_coverage_gain': stack_gains,
                                        'pose_i' : pose_i
                                    }
                                    store_experience(db_env, experience_db)


                                    
                        experiences_list = []    
                        
                    Dijkstra_path = []
                    path_record = 0
                    

                    current_model_input = torch.cat((current_pc_imgs, current_previous_trajectory_img), dim=1).to(device)
                    predicted_value_map, _ = nbp(current_model_input)

                    max_gain_map, _ = torch.max(predicted_value_map, dim=1, keepdim=True)
                    camera_position_value_list = []

                    # Iterate through all possible camera positions, calculate best value for each position
                    for key, point_3d in splited_pose_space.items():
                        if camera.cam_idx[:3].tolist() != ast.literal_eval(key):
                            point_2d = transform_points_to_n_pieces(point_3d.unsqueeze(0), camera_current_pose, device) # tensor([[[52.8558, 38.0611]]], device='cuda:1')
                            grid_position = get_point_position_in_the_img(point_2d.squeeze(0), value_map_size, prediction_range)
                            
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
                    gain_values = torch.stack([item[2] for item in camera_position_value_list])
                    beta = 0.5  # temperature parameter
                    probabilities = torch.softmax(gain_values / beta, dim=0)
                    selected_idx = torch.multinomial(probabilities, 1).item()
                    selected_item = camera_position_value_list.pop(selected_idx)
                    camera_position_value_list.insert(0, selected_item)

                    path_start_position = camera.cam_idx[:3].tolist()
                    for pose_location in camera_position_value_list:

                        path_end_position = ast.literal_eval(pose_location[0])
                        if check_camera_in_mesh(mesh_for_check, splited_pose_space[pose_location[0]]):

                            if path_end_position in unreachable_position:
                                continue
                            
                            Dijkstra_path = generate_Dijkstra_path(splited_pose_space, path_start_position, path_end_position, mesh_for_check, camera_current_pose, camera,
                                                                    value_map_size, prediction_range, predicted_value_map, device)
                            if Dijkstra_path is not None:
                                experiences_list.append([
                                        coverage_evolution[-1],
                                        current_model_input,
                                        current_gt_obs,
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
                                        current_gt_obs,
                                        camera_current_pose,
                                        camera.cam_idx[-1]
                                    ])


                if Dijkstra_path is None:
                    break
                

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
                    
                # Depth gain
                all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
                                camera=camera,
                                n_frames=params.n_interpolation_steps,
                                n_alpha=params.n_alpha_for_supervision,
                                return_gt_zbuf=True)
                
                batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=device, all_zbuf=all_zbuf)
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


