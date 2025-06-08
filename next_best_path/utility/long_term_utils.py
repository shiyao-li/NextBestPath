import torch
import copy
import time
import math
import heapq
import statistics
import random
import json
import torch.nn.functional as F
import matplotlib.colors as mcolors
from macarons.utility.render_utils import plot_point_cloud
from functools import reduce
import matplotlib.pyplot as plt
from macarons.utility.macarons_utils import *
from macarons.utility.utils import count_parameters
from macarons.testers.scene import *
from macarons.utility.render_utils import plot_point_cloud, plot_graph
from torch.nn.functional import pairwise_distance
from macarons.trainers.train_macarons import recompute_mapping
from next_best_path.utility.utils import *

def setup_nbp_test(params, model_path, device, verbose=True):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path)
    print("\nThe following scenes will be used to test the model:")
    for batch, elem in enumerate(test_dataloader):
        print(elem['scene_name'][0])

    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0, verbose=verbose)

    return test_dataloader, memory


def obtain_depth(params, batch_dict, alpha_dict, device,
                      depth_loss_fn=None, pose_loss_fn=None, regularity_loss_fn=None, ssim_loss_fn=None,
                      compute_loss=False,
                      use_perfect_depth=False):

    images = 0. + batch_dict['images']
    mask = False + batch_dict['mask'].bool()
    R = 0. + batch_dict['R']
    T = 0. + batch_dict['T']
    zfar = batch_dict['zfar']

    alpha_images = 0. + alpha_dict['images']
    alpha_mask = False + alpha_dict['mask'].bool()
    alpha_R = 0. + alpha_dict['R']
    alpha_T = 0. + alpha_dict['T']
    alpha_zfar = alpha_dict['zfar']

    if use_perfect_depth:
        if ('zbuf' not in batch_dict) or ('zbuf' not in alpha_dict):
            raise NameError("Parameter use_perfect_depth is True but no zbuf is provided in input dictionaries.")

    batch_size = images.shape[0]

    # Initialize prediction and ground truth tensors

    x = transpose_channels(images, channel_is_at_the_end=True)
    x_alpha = transpose_channels(alpha_images, channel_is_at_the_end=True)

    # Changing camera poses to make them relative to initial frame
    alpha_R, alpha_T = get_relative_pose_matrices(R, alpha_R, T, alpha_T)
    R = torch.eye(n=3).view(1, 3, 3).expand(batch_size, -1, -1).to(device)
    T = torch.zeros_like(T).to(device)

    symmetry_applied = False
    if params.data_augmentation:
        coin_flip = np.random.rand()
        if coin_flip < params.jitter_probability:
            x, x_alpha = apply_jitter_to_images(params, x, x_alpha)

        coin_flip = np.random.rand()
        if coin_flip < params.symmetry_probability:
            symmetry_applied = True
            x, _, R, T, mask = apply_symmetry_to_images(x=x, zbuf=None, R=R, T=T, mask=mask)
            x_alpha, _, alpha_R, alpha_T, alpha_mask = apply_symmetry_to_images(x=x_alpha,
                                                                                zbuf=None,
                                                                                R=alpha_R,
                                                                                T=alpha_T,
                                                                                mask=alpha_mask)
        images = transpose_channels(x, channel_is_at_the_end=False)
        alpha_images = transpose_channels(x_alpha, channel_is_at_the_end=False)

    # Computing gt_pose and corresponding gt_factor
    gt_pose = convert_matrix_to_pose(params, R, T, alpha_R, alpha_T)

    # Prediction
    if use_perfect_depth:
        zbuf = transpose_channels(batch_dict['zbuf'], channel_is_at_the_end=True)
        zbuf = torch.clamp(zbuf, min=params.znear, max=params.zfar)
        pose = 0. + gt_pose[:, :params.n_alpha]
        depth1 = 0. + zbuf
        depth2 = 0. + zbuf
        depth3 = 0. + zbuf
        depth4 = 0. + zbuf

        disp1 = compute_disparity_from_depth(params, depth1)


    # We compute the masks if needed
    if params.use_depth_mask:
        mask1 = transpose_channels(mask, channel_is_at_the_end=True).bool()
        mask2 = mask1  # transpose_channels(batch_mask, channel_is_at_the_end=True)
        mask3 = mask1  # transpose_channels(batch_mask, channel_is_at_the_end=True)
        mask4 = mask1  # transpose_channels(batch_mask, channel_is_at_the_end=True)
    else:
        mask1 = None
        mask2 = None
        mask3 = None
        mask4 = None

    transposed_images = transpose_channels(images, channel_is_at_the_end=True)

    # Compute error mask
    with torch.no_grad():
        norm_disp1 = 0. + disp1.detach()
        mean_disp1 = norm_disp1.mean(2, True).mean(3, True)
        norm_disp1 = norm_disp1 / (mean_disp1 + 1e-7)
        norm_disp1[~mask1] *= 0.
        error_tab = regularity_tab(disp=pad(norm_disp1.detach(), padding=1, padding_mode='reflect'),
                                   img=pad(transposed_images, padding=1, padding_mode='reflect'))
        error_threshold = error_tab.view(batch_size, -1).mean(dim=-1) + error_tab.view(batch_size, -1).std(dim=-1)
        error_threshold = error_threshold.view(batch_size, 1, 1, 1)  # .expand(-1, -1, params.image_height, params.image_width)
        error_mask = error_tab < error_threshold


    depth = depth1.detach()
    mask = mask1.detach()
    if symmetry_applied:
        depth = hflip(depth)
        mask = hflip(mask)
        error_mask = hflip(error_mask)


    return transpose_channels(depth, channel_is_at_the_end=False), \
            transpose_channels(mask, channel_is_at_the_end=False).bool(), \
            transpose_channels(error_mask, channel_is_at_the_end=False).bool(), \
            pose.detach(), gt_pose.detach()


def check_camera_in_mesh(mesh_for_check, camera_position):
    ray_directions_1 = np.array([[0, 1, 0]])
    ray_directions_2 = np.array([[1, 0, 0]])
    ray_directions_3 = np.array([[0, 0, 1]])
    inter_locations_1, _, _ = mesh_for_check.ray.intersects_location(ray_origins=[camera_position.cpu().numpy()],
                                                                        ray_directions=ray_directions_1)
    
    inter_locations_2, _, _ = mesh_for_check.ray.intersects_location(ray_origins=[camera_position.cpu().numpy()],
                                                                        ray_directions=ray_directions_2)
    inter_locations_3, _, _ = mesh_for_check.ray.intersects_location(ray_origins=[camera_position.cpu().numpy()],
                                                                        ray_directions=ray_directions_3)
    
    return (len(inter_locations_1) % 2 == 1) and (len(inter_locations_2) % 2 == 1) and (len(inter_locations_3) % 2 == 1)


#====================Dijkstra implementation========================

def generate_Bidirectional_Dijkstra_path(pose_space, start_position, end_position, scene, device):

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y+1, z],
            [x, y-1, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        print(pose_space)
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                     and not line_segment_intersects_point_cloud_region(scene, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
        return neighbors

    start, goal = tuple(start_position), tuple(end_position)
    
    frontier_start, frontier_goal = [(0, start)], [(0, goal)]
    
    came_from_start, came_from_goal = {start: None}, {goal: None}
    cost_so_far_start, cost_so_far_goal = {start: 0}, {goal: 0}
    
    meet_point = None
    
    while frontier_start and frontier_goal:
        current_cost_start, current_start = heapq.heappop(frontier_start)
        current_cost_goal, current_goal = heapq.heappop(frontier_goal)

        if current_start in cost_so_far_goal or current_goal in cost_so_far_start:
            meet_point = current_start if current_start in cost_so_far_goal else current_goal
            break
        
        for next_start in get_neighbors(list(current_start)):
            next_tuple_start = tuple(next_start)
            new_cost_start = cost_so_far_start[current_start] + 1
            if next_tuple_start not in cost_so_far_start or new_cost_start < cost_so_far_start[next_tuple_start]:
                cost_so_far_start[next_tuple_start] = new_cost_start
                heapq.heappush(frontier_start, (new_cost_start, next_tuple_start))
                came_from_start[next_tuple_start] = current_start

        for next_goal in get_neighbors(list(current_goal)):
            next_tuple_goal = tuple(next_goal)
            new_cost_goal = cost_so_far_goal[current_goal] + 1
            if next_tuple_goal not in cost_so_far_goal or new_cost_goal < cost_so_far_goal[next_tuple_goal]:
                cost_so_far_goal[next_tuple_goal] = new_cost_goal
                heapq.heappush(frontier_goal, (new_cost_goal, next_tuple_goal))
                came_from_goal[next_tuple_goal] = current_goal

    if meet_point:
        path_start, path_goal = [], []

        current = meet_point
        while current:
            path_start.append(list(current))
            current = came_from_start.get(current)

        current = came_from_goal[meet_point]
        while current:
            path_goal.append(list(current))
            current = came_from_goal.get(current)

        # Combine paths and ensure no collisions
        full_path = path_start[::-1] + path_goal
        collision_free_path = [full_path[0]]

        for i in range(1, len(full_path)):
            if not line_segment_intersects_point_cloud_region(scene, pose_space[str(full_path[i-1]).replace(", ", ",  ")], pose_space[str(full_path[i]).replace(", ", ",  ")], device):
                collision_free_path.append(full_path[i])
            else:
                # Collision detected; the path is invalid
                print("Path has collision!")
                return []

        real_move_path = [pose_space[str(idx).replace(", ", ",  ")] for idx in collision_free_path]
        return real_move_path[1:]
    else:
        print("No valid path found!")
        return []
    
def move_a_random_step(pose_space, start_position, scene, occ_probs, device):
    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y+1, z],
            [x, y-1, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                    and not line_segment_intersects_point_cloud_region(scene, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
                    # and not predict_collision_with_occupancy_field(scene, occ_probs, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
        return neighbors
    start = tuple(start_position)
    neighbors = get_neighbors(list(start))
    return [random.choice(neighbors)]

    

def bresenham_line(x0, y0, x1, y1):
    """生成两点之间的线段上的所有像素点（使用Bresenham's Line Algorithm）"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def line_across_image_pixel(point_1, point_2, camera_current_pose, grid_size, grid_range, layout_image, device):
    point_1_2d = transform_points_to_n_pieces(point_1.unsqueeze(0), camera_current_pose, device)
    point_2_2d = transform_points_to_n_pieces(point_2.unsqueeze(0), camera_current_pose, device)
    grid_position_1 = get_point_position_in_the_img(point_1_2d.squeeze(0), grid_size, grid_range)
    grid_position_2 = get_point_position_in_the_img(point_2_2d.squeeze(0), grid_size, grid_range)
    if 0 <= grid_position_1[0] < grid_size[0] and 0 <= grid_position_1[1] < grid_size[1]:
        if 0 <= grid_position_2[0] < grid_size[0] and 0 <= grid_position_2[1] < grid_size[1]:
            point_1_img_position = torch.tensor([grid_position_1[0], grid_position_1[1]]).to(device)
            point_2_img_position = torch.tensor([grid_position_2[0], grid_position_2[1]]).to(device)
            point_1_img_position = point_1_img_position.round().long()
            point_2_img_position = point_2_img_position.round().long()
            
            # Extract the 2D layout image from the tensor
            layout_img_2d = layout_image[0, 0]  
            
            # Get the line pixels using Bresenham's algorithm or similar
            line_pixels = bresenham_line(point_1_img_position[0].item(), point_1_img_position[1].item(),
                                         point_2_img_position[0].item(), point_2_img_position[1].item())

            # Check if any pixel in the line is set to 1 in the layout image
            obstacle_count = 0
            for (x, y) in line_pixels:
                if layout_img_2d[x, y].item() == 1:
                    obstacle_count += 1
            if obstacle_count >= 2:
                return True  # The line passes through at least 4 consecutive obstacle pixels
            
            return False 
        else:
            return True
    else: 
        return True


def generate_Dijkstra_path(pose_space, start_position, end_position, mesh_for_check, camera_current_pose, camera, grid_size, grid_range, 
                              gain_map_prediction, device, layout_image=None, layout_size=None, collision_list=None, training_flag=True, passable_list=None):

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        if training_flag:
            neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                        and not line_segment_mesh_intersection(pose_space[str(position).replace(", ", ",  ")][:3], pose_space[str(n).replace(", ", ",  ")][:3], mesh_for_check)]

        else:
            neighbors = [n for n in potential_neighbors 
                 if str(n).replace(", ", ",  ") in pose_space  
                 and ([position, n] in passable_list 
                      or (not line_across_image_pixel(pose_space[str(position).replace(", ", ",  ")][:3], 
                                                          pose_space[str(n).replace(", ", ",  ")][:3], 
                                                          camera_current_pose, layout_size, grid_range, 
                                                          layout_image, device)  # avoid collision by using layout image
                          and [position, n] not in collision_list))]  

        return neighbors

    start = tuple(start_position)
    goal = tuple(end_position)

    frontier = []  
    heapq.heappush(frontier, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in get_neighbors(list(current)):
            next_tuple = tuple(next)
            new_cost = cost_so_far[current] + 1 
            if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                cost_so_far[next_tuple] = new_cost
                heapq.heappush(frontier, (new_cost, next_tuple))
                came_from[next_tuple] = current

    if goal in came_from: 
        path = []
        real_move_path = []
        current = goal
        while current:
            path.append(list(current))
            current = came_from[current]
        path.reverse()
        for idx in path:
            real_move_path.append(idx)
        real_move_path_tesnor = []
        for step in real_move_path:
            location_tensor = pose_space[str(step).replace(", ", ",  ")]
            location_2d = transform_points_to_n_pieces(location_tensor.unsqueeze(0), camera_current_pose, device)
            grid_location = get_point_position_in_the_img(location_2d.squeeze(0), grid_size, grid_range)

            if 0 <= grid_location[0] < grid_size[0] and 0 <= grid_location[1] < grid_size[1]:
                values, indices = gain_map_prediction[0, :, grid_location[0], grid_location[1]].sort(descending=True)
                for idx in indices.tolist():
                    step_tensor = torch.tensor([step[0], step[1], step[2], 2, idx]).to(device)
                    if not torch.any(torch.all(camera.cam_idx_history == step_tensor, dim=1)).item():
                        break
            else:
                while True:
                    gain_map_idx = random.randint(0, 7)
                    step_tensor = torch.tensor([step[0], step[1], step[2], 2, gain_map_idx], device=device)
                    if not torch.any(torch.all(camera.cam_idx_history == step_tensor, dim=1)).item():
                        break

            real_move_path_tesnor.append(step_tensor)

        real_move_path_tesnor = torch.stack(real_move_path_tesnor)
        return real_move_path_tesnor[1:, :]
    else:
        return None

def generate_key_value_splited_dict(orig_dict):
    '''
    Generate a splited dictionary:{position_index, real_position_camera}
    '''
    new_dict = {}

    for key, value in orig_dict.items():
        new_key = key.split(",")[0:3]
        new_key = ", ".join(new_key) + "]"
        
        new_value = value[0]
        
        new_dict[new_key] = new_value
    return new_dict


#====================Useful functions========================
def random_sample_pc(pc, num_samples):
    """
    Randomly samples the point cloud to the specified number of samples.
    If the point cloud has fewer points than the specified number of samples, the original point cloud is returned.
    """
    n_points = pc.shape[0]
    if n_points <= num_samples:
        return pc
    else:
        indices = torch.randperm(n_points)[:num_samples]
        return pc[indices]

def find_nearest_points_distances(pc1, pc2):
    """
    Manually calculate the distance from each point in pc1 to the nearest point in pc2.
    """
    distances = torch.cdist(pc1, pc2, p=2)  # Calculate the Euclidean distance between all pairs of points
    min_distances, _ = torch.min(distances, dim=1)  # Find the nearest point and its distance
    return min_distances

def calculate_coverage_percentage(pc1, pc2, threshold=1, weight=2):
    """
    Calculate the percentage of similar points that satisfy the distance threshold.
    """
    if len(pc2) == 0:
        return 0.
    else:
        # Randomly sample pc2 so that it has the same number of points as pc1
        sampled_pc2 = random_sample_pc(pc2, int(len(pc1)*weight))
        nearest_distances = find_nearest_points_distances(pc1, sampled_pc2)
        similar_points = (nearest_distances < threshold).float().mean() 
        return similar_points.item()

def calculate_fullpc_coverage(base_cloud, rebuild_cloud, threshold=1):
    if len(rebuild_cloud) == 0:
        return 0
    else:
        distances = torch.cdist(base_cloud, rebuild_cloud, p=2.0)
    
        # 找到每个基准点到重建点云中最近点的距离
        min_distances = distances.min(dim=1).values
        
        # 计算在阈值内的距离数量
        coverage_count = (min_distances <= threshold).sum().item()
        
        # 计算覆盖率百分比
        total_base_points = base_cloud.shape[0]
        coverage_percentage = (coverage_count / total_base_points) * 100
        
        return coverage_percentage
    
def compute_auc(y, dx=1/40):
   auc = np.trapz(y, dx=dx) + y[0] * dx / 2.
   return auc

def gradient_colormap(values, device='cpu'):
    """
    Maps values to a gradient of colors from light to dark with more distinct change.
    :param values: A one-dimensional torch.Tensor containing the values, normalized between 0 and 1.
    :param device: Device where computations will be performed.
    :return: An N x 3 torch.Tensor containing RGB colors.
    """
    # Ensure values are on the correct device
    values = values.to(device)

    # Modify the color gradient for more distinct change
    # Adjusting the gradient to be more distinct by using a non-linear transformation
    values_transformed = torch.pow(values, 0.5) # Non-linear transformation for more distinct color changes
    values_transformed = values_transformed.unsqueeze(-1) 
    # Define the color gradient from blue (0.0, 0.0, 1.0) to red (1.0, 0.0, 0.0)
    start_color = torch.tensor([0.0, 0.0, 1.0], device=device)  # Blue
    end_color = torch.tensor([1.0, 0.0, 0.0], device=device)    # Red

    # Linearly interpolate colors based on transformed values
    colors = start_color * (1 - values_transformed) + end_color * values_transformed

    return colors.squeeze()

def downsample_point_cloud(point_cloud, num_points=100000):
    """
    Randomly downsamples a point cloud to a specified number of points and rounds coordinates to two decimal places.

    Parameters:
    point_cloud (torch.Tensor): The original point cloud, shape (N, 3).
    num_points (int): Number of points to downsample to.

    Returns:
    torch.Tensor: Downsampled point cloud with coordinates rounded to two decimal places, shape (num_points, 3).
    """
    # Get the number of original points
    original_num_points = point_cloud.shape[0]

    # Downsample the point cloud
    if original_num_points > num_points:
        indices = torch.randperm(original_num_points)[:num_points]
        downsampled_point_cloud = point_cloud[indices]
    elif original_num_points < num_points:
        extra_indices = torch.randint(0, original_num_points, (num_points - original_num_points,))
        downsampled_point_cloud = torch.cat((point_cloud, point_cloud[extra_indices]), dim=0)
    else:
        downsampled_point_cloud = point_cloud

    return downsampled_point_cloud

def compute_bounding_box_tensor(point_cloud_tensor):
    """
    Computes the bounding box for a given point cloud stored as a PyTorch tensor.

    Parameters:
    point_cloud_tensor (torch.Tensor): A tensor of shape (N, 3) where each row represents a point in 3D space.

    Returns:
    tuple: A tuple containing two tuples, each with three elements. 
           The first tuple represents the minimum (x, y, z) coordinates, 
           and the second tuple represents the maximum (x, y, z) coordinates.
    """
    min_coords, _ = torch.min(point_cloud_tensor, dim=0)
    max_coords, _ = torch.max(point_cloud_tensor, dim=0)

    return min_coords, max_coords

def compute_coverage_gain_for_path(mcts_visibility_gains_list):
    # (neighbor_idx, coverage_gain, visibility_gains, fov_occ_probs, fov_X_mask, sample_idx, fov_X_world)
    all_fov_X_world = {}
    
    valid_items = [item for item in mcts_visibility_gains_list if item[-4] is not None]
    
    for _, _, _, fov_occ_probs, _, _, fov_X_world in valid_items:
        for i in range(fov_X_world.size(0)):
            key = tuple(fov_X_world[i].tolist())
            all_fov_X_world[key] = max(all_fov_X_world.get(key, 0), fov_occ_probs[i].item())
    union_volume = sum(all_fov_X_world.values())
    
    all_points = torch.cat([item[6][item[5]] for item in valid_items], dim=0)
    all_gains = torch.cat([item[2].squeeze() for item in valid_items], dim=0)
    
    _, unique_indices = torch.unique(all_points, dim=0, return_inverse=True)
    
    max_gains = torch.zeros_like(unique_indices, dtype=torch.float)
    max_gains.scatter_add_(0, unique_indices, all_gains)

    combined_visibility_gains = max_gains.unsqueeze(0).unsqueeze(1)
    coverage_gain = torch.mean(combined_visibility_gains, dim=-1) * union_volume
    return coverage_gain.view(-1, 1).item()


def predict_path_coverage(visibility_gains_list):
    # One item in visibility_gains_list should be like (visibility_gain, fov_proxy_volume)
    if len(visibility_gains_list) != 8:
        print("God, please check the simulation")
    else:
        visibility_tensor = [visibility_gains_list[0][0], visibility_gains_list[1][0], visibility_gains_list[2][0], visibility_gains_list[3][0], visibility_gains_list[4][0], visibility_gains_list[5][0], visibility_gains_list[6][0], visibility_gains_list[7][0]]
        max_visibility = torch.max(torch.cat(visibility_tensor, dim=0), dim=0).values
        coverage_gain = torch.mean(max_visibility, dim=-1) * visibility_gains_list[7][1]
        return coverage_gain[0].item()

def predict_collision_with_occupancy_field(X_world, occ_probs, start_point, end_point, device, threshold_distance=1):
    """
    X_world: torch.Tensor, shape [n, 3] - The point cloud in world coordinates
    occ_probs: torch.Tensor, shape [n, 1] - The occupancy probability for each point in the point cloud
    start_point, end_point: torch.Tensor, shape [3] - The start and end points of the line segment
    device: torch.device - The device (CPU or GPU) where tensors should be placed
    threshold_distance: float - The threshold distance for considering a point as intersecting
    
    Returns: bool - Whether the line segment intersects with a region of the point cloud
    """
    
    # Ensure tensors are on the correct device
    # X_world = X_world.to(device)
    # occ_probs = occ_probs.to(device)
    # start_point = start_point.to(device)
    # end_point = end_point.to(device)
    
    line_vector = end_point - start_point
    # Calculate the vectors from points to the line segment start and end points
    point_to_start = X_world - start_point
    point_to_end = X_world - end_point
    
    # Calculate the projection vectors from points to the line segment start point
    projection_vector = torch.sum(point_to_start * line_vector, dim=1) / torch.norm(line_vector)**2
    
    # Calculate the distances from points to the line segment
    distances = torch.zeros(X_world.shape[0]).to(device)
    
    # Create a mask to identify points within the segment range
    within_segment_mask = (projection_vector >= 0) & (projection_vector <= 1)
    
    # Calculate distances for points within the segment range
    if within_segment_mask.any():
        closest_point = start_point + projection_vector[within_segment_mask].unsqueeze(1) * line_vector
        distances_within_segment = torch.norm(X_world[within_segment_mask] - closest_point, dim=1)
        distances[within_segment_mask] = distances_within_segment
    
    # Calculate distances for points outside the segment range
    distances[~within_segment_mask] = torch.min(torch.norm(point_to_start[~within_segment_mask], dim=1),
                                                torch.norm(point_to_end[~within_segment_mask], dim=1))
    
    # Find points within the threshold distance
    close_points_mask = distances < threshold_distance
    
    # If there are any close points
    if close_points_mask.any():
        close_point_probs = occ_probs[close_points_mask]
        # Check if any close points have significant occupancy probability
        # or all close points have exactly 0.5 occupancy probability
        # Note: You might adjust "significant_probability_threshold" as per your use case
        significant_probability_threshold = 0.9
        # if (torch.max(close_point_probs).item() >= significant_probability_threshold):
        #     print(1111)
        count_above_threshold = (close_point_probs > significant_probability_threshold).sum().item()
        if count_above_threshold >= 5:
            return True
    return False

def line_segment_intersects_point_cloud_region(scene, start_point, end_point, device):
    # Retrieve the entire point cloud from the scene
    # point_cloud = scene.return_entire_pt_cloud(return_features=False)
    point_cloud = scene
    line_vector = end_point - start_point

    # Check the line segment length to prevent division by a very small number
    line_length_squared = torch.norm(line_vector)**2
    # if line_length_squared < 1e-8:  # Use a small threshold, e.g., 1e-8
    #     return False

    # Calculate vectors from each point in the cloud to the start and end of the line segment
    point_to_start = point_cloud - start_point
    point_to_end = point_cloud - end_point

    # Compute projection vectors using vectorized operations
    projection_vector = torch.sum(point_to_start * line_vector, dim=1) / line_length_squared

    # Create a mask to identify points within the line segment range
    within_segment_mask = (projection_vector >= 0) & (projection_vector <= 1)

    # Compute closest points on the line segment for points within the segment range
    closest_point = start_point + projection_vector.unsqueeze(1) * line_vector
    distances_within_segment = torch.norm(point_cloud - closest_point, dim=1)

    # Compute the minimum distance to the line segment endpoints for points outside the segment range
    distances_outside_segment = torch.min(torch.norm(point_to_start, dim=1), torch.norm(point_to_end, dim=1))

    # Combine distances for both cases
    distances = torch.where(within_segment_mask, distances_within_segment, distances_outside_segment)

    # Check for intersection
    if torch.min(distances).item() < 0.2:  # Intersection threshold
        return True
    return False


def calculate_pose_distance(T1, T2):
    # T1, T2: Translation 
    translation_distance = torch.norm(T1 - T2)
    distance = torch.sqrt(translation_distance**2)
    return distance

