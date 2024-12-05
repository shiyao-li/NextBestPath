import torch
import copy
import time
import math
import heapq
import statistics
import random
import json
import concurrent.futures
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
from navi.utility.utils import *
from torch_geometric.nn import knn


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


def generate_Dijkstra_path_2d_occ(pose_space, start_position, end_position, X_world, occ_probs, camera_current_pose, camera, grid_size, grid_range, gain_map_prediction, device, collision_list):

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                    and not predict_collision_with_occupancy_field(X_world, occ_probs, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)
                    and [position, n] not in collision_list]
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
        # ------修改为增加一个最大index
        for step in real_move_path:
            location_tensor = pose_space[str(step).replace(", ", ",  ")]
            location_2d = batch_transform_points_n_pieces(location_tensor.unsqueeze(0), camera_current_pose, device)
            grid_location = get_grid_position(location_2d.squeeze(0), grid_size, grid_range)

            # 检查索引是否越界，如果越界则使用 previous_gain_map_idx
            if 0 <= grid_location[0] < grid_size[0] and 0 <= grid_location[1] < grid_size[1]:
                values, indices = gain_map_prediction[0, :, grid_size[0] - grid_location[1] - 1, grid_location[0]].sort(descending=True)
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
    point_1_2d = batch_transform_points_n_pieces(point_1.unsqueeze(0), camera_current_pose, device)
    point_2_2d = batch_transform_points_n_pieces(point_2.unsqueeze(0), camera_current_pose, device)
    grid_position_1 = get_grid_position(point_1_2d.squeeze(0), grid_size, grid_range)
    grid_position_2 = get_grid_position(point_2_2d.squeeze(0), grid_size, grid_range)
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


def generate_Dijkstra_path_2d(pose_space, start_position, end_position, mesh_for_check, camera_current_pose, camera, grid_size, grid_range, 
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
            location_2d = batch_transform_points_n_pieces(location_tensor.unsqueeze(0), camera_current_pose, device)
            grid_location = get_grid_position(location_2d.squeeze(0), grid_size, grid_range)

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
    
def generate_Dijkstra_path_optimized(pose_space_keys, start_position, end_position, mesh_for_check, device):
      # Preprocess keys to tuple

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            (x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)
        ]
        neighbors = [
            n for n in potential_neighbors if n in pose_space_keys
            and not line_segment_mesh_intersection(pose_space_keys[position][:3], pose_space_keys[n][:3], mesh_for_check)
        ]
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

        for next in get_neighbors(current):
            new_cost = cost_so_far[current] + 1  # Since every step has a cost of 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                heapq.heappush(frontier, (new_cost, next))
                came_from[next] = current

    if goal in came_from:
        path = []
        real_move_path = []
        current = goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
        for idx in path:
            real_move_path.append(pose_space_keys[idx])
        return real_move_path[1:]
    else:
        return None


def generate_Dijkstra_path(pose_space, start_position, end_position, mesh_for_check, device):

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            # [x, y+1, z],
            # [x, y-1, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                    # and not line_segment_intersects_point_cloud_region(scene, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
                    # and not predict_collision_with_occupancy_field(scene, occ_probs, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
                    and not line_segment_mesh_intersection(pose_space[str(position).replace(", ", ",  ")][:3], pose_space[str(n).replace(", ", ",  ")][:3], mesh_for_check)]
        return neighbors

    start = tuple(start_position)
    goal = tuple(end_position)

    frontier = []  # Using a priority queue with (cost, position)
    heapq.heappush(frontier, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in get_neighbors(list(current)):
            next_tuple = tuple(next)
            new_cost = cost_so_far[current] + 1  # Since every step has a cost of 1
            if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                cost_so_far[next_tuple] = new_cost
                heapq.heappush(frontier, (new_cost, next_tuple))
                came_from[next_tuple] = current

    # reconstruct the path
    if goal in came_from:  # ensure there is a path
        path = []
        real_move_path = []
        current = goal
        while current:
            path.append(list(current))
            current = came_from[current]
        path.reverse()
        for idx in path:
            real_move_path.append(pose_space[str(idx).replace(", ", ",  ")])
        return real_move_path[1:]
    else:
        return None


def generate_pro_Dijkstra_path(pose_space, start_position, end_position, scene, device, history, mesh_for_check):

    def get_neighbors(position):
        x, y, z = position
        potential_neighbors = [
            [x+1, y, z],
            [x-1, y, z],
            [x, y, z+1],
            [x, y, z-1]
        ]
        neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                     # and not line_segment_intersects_point_cloud_region(scene, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
                     and not line_segment_mesh_intersection(pose_space[str(position).replace(", ", ",  ")][:3], pose_space[str(n).replace(", ", ",  ")][:3], mesh_for_check)]
        return neighbors
    
    def is_in_history(point):
        real_value = pose_space[str(list(point)).replace(", ", ",  ")]
        return torch.any(torch.all(history == real_value, dim=1))

    # Added function to check if the point is in history

    start = tuple(start_position)
    goal = tuple(end_position)

    frontier = []  # Using a priority queue with (cost, position)
    heapq.heappush(frontier, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in get_neighbors(list(current)):
            next_tuple = tuple(next)
            penalty = 10 if is_in_history(next_tuple) else 0  # You can adjust the penalty value as needed
            new_cost = cost_so_far[current] + 1 + penalty
            if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                cost_so_far[next_tuple] = new_cost
                heapq.heappush(frontier, (new_cost, next_tuple))
                came_from[next_tuple] = current

    # reconstruct the path
    if goal in came_from:  # ensure there is a path
        path = []
        real_move_path = []
        current = goal
        while current:
            path.append(list(current))
            current = came_from[current]
        path.reverse()
        for idx in path:
            real_move_path.append(torch.tensor(idx).to(device))
            # real_move_path.append(pose_space[str(idx).replace(", ", ",  ")])
        return real_move_path[1:]
    else:
        print("No valid path found!")
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

#====================RRT implementation========================

def generate_rrt_path(pose_space, start_position, scene, device):
    
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
        
        valid_neighbors = [n for n in potential_neighbors if str(n).replace(", ", ",  ") in pose_space 
                 and not line_segment_intersects_point_cloud_region(scene, pose_space[str(position).replace(", ", ",  ")], pose_space[str(n).replace(", ", ",  ")], device)]
        
        return valid_neighbors

    visited = set()
    stack = [start_position]
    path = [start_position]
    
    while len(path) < 101:
        if not stack:
            break

        current = stack[-1]

        neighbors = get_neighbors(current)
        unvisited_neighbors = [n for n in neighbors if tuple(n) not in visited]
        
        # Randomly pick an unvisited neighbor or a visited one if none unvisited exist
        if unvisited_neighbors:
            next_step = random.choice(unvisited_neighbors)
        else:
            next_step = random.choice(neighbors) if neighbors else None

        if next_step:
            if tuple(next_step) not in visited:
                visited.add(tuple(next_step))
            stack.append(next_step)
            path.append(next_step)
        else:  
            stack.pop()

    return path[1:]


def old_generate_rrt_path(pose_space, start_position, scene, max_steps, device):
    '''
    Generate a path starting from a given start_position by using RRT method

    :param pose_space: 
    :param start_position:
    :param max_step:
    '''
    path = [start_position]
    positions = torch.cat((pose_space[1:], start_position), dim=0)

    # 设置轴的概率，使x和z轴的概率高于y轴的概率
    axis_probabilities = [0.3, 0.3, 0.3]
    i = 0
    while i < max_steps+1:
        axis_choice = random.choices([0, 1, 2], weights=axis_probabilities, k=1)[0]
        
        # 对于非选择的轴，创建一个布尔掩码，该掩码只有在点的位置与当前点的位置相同时才为真
        mask = torch.ones(positions.shape[0], dtype=torch.bool, device=device)
        for other_axis in set(range(3)) - {axis_choice}:
            mask &= torch.eq(positions[:, other_axis], path[-1][0, other_axis])

        # 将掩码应用到我们的点集，以便我们只计算符合要求的点的距离
        relevant_positions = positions[mask]
        all_intersections = all(line_segment_intersects_point_cloud_region(scene, path[-1], p.unsqueeze(0), device) for p in relevant_positions)

        # If there is no candidate positions in the pool, we can move back
        if relevant_positions.shape[0] == 0 or all_intersections:
            if len(path) > 2:
                # Append the second last point in the path to the end of the path
                path.append(path[-2])  
                i += 1
                continue
            else:
                # No valid points left and no previous points to go back to, break the loop
                break
        distances = (relevant_positions[:, axis_choice] - path[-1][0, axis_choice])**2
        closest_point_index_in_masked = torch.argmin(distances).item()

        # Find the corresponding index in the original positions tensor
        closest_point_index = torch.where(mask)[0][closest_point_index_in_masked]
        closest_point = positions[closest_point_index].unsqueeze(0)
        if not line_segment_intersects_point_cloud_region(scene, path[-1], closest_point, device):
            path.append(closest_point)
            # Remove the closest_point from positions
            positions = torch.cat((positions[:closest_point_index], positions[closest_point_index+1:]), dim=0)
            i += 1
    path = path[2: ]
    return path

# def count_duplicates(path):
#     xz_counts = {}

#     for point in path:
#         xz = (point[0], point[2])

#         if xz in xz_counts:
#             xz_counts[xz] += 1
#         else:
#             xz_counts[xz] = 1

#     return sum(1 for count in xz_counts.values() if count > 3)

# def count_y_moves(path):
#     y_moves_count = 0

#     for i in range(1, len(path)):
#         prev_y = path[i - 1][1]
#         current_y = path[i][1]

#         if abs(current_y - prev_y) > 1e-5:  # Threshold to consider a y-axis move
#             y_moves_count += 1
#     return y_moves_count

def count_valid_y_axis_rows(path):
    points_dict = {}

    for point in path:
        x, z = point[0], point[2]
        if x in points_dict:
            points_dict[x].add(z)
        else:
            points_dict[x] = set([z])

    # 计算每个x下有多少个不同的点
    count_dict = {x: len(z_set) for x, z_set in points_dict.items()}

    # 获取value大于2的x坐标集合
    x_values_gt_2 = sorted([x for x, value in count_dict.items() if value > 2])
    
    # Step 1: 获取value大于3的x坐标集合
    x_values_gt_3 = [x for x in x_values_gt_2 if count_dict[x] > 6]

    # Step 2: 计算相邻两点之间的距离
    distances = [((x_values_gt_3[i+1] - x_values_gt_3[i])**2) for i in range(len(x_values_gt_3)-1)]

    # Step 3: 计算距离的方差
    variance = statistics.variance(distances) if len(distances) > 1 else 0
    return sum(1 for value in count_dict.values() if value > 3), variance


def find_optimal_path(all_paths, repeat_weight=0.3):
    max_y_axis = float('-inf')
    max_y_axis_paths = []  # Store paths that has max y_axis value

    for path in all_paths:
        xz_num, distance = count_valid_y_axis_rows(path)
        score = distance
        
        if xz_num > max_y_axis:
            max_y_axis = xz_num
            max_y_axis_paths = [(path, score)]  # New max_y_axis value found, refresh the list
        elif xz_num == max_y_axis:
            max_y_axis_paths.append((path, score))  # Same max_y_axis value, append to the list

    # At this point, max_y_axis_paths contains all paths with maximum y-axis count and their respective scores.
    # Now, let's find the path with minimal score among these.
    optimal_path = min(max_y_axis_paths, key=lambda x: x[1])[0]  # Get the path part from the (path, score) tuple

    return optimal_path

#====================MCTS implementation========================

class ExploreState:
    def __init__(self, mesh, params, macarons, camera, full_pc, full_pc_colors, surface_scene,
                 covered_scene, proxy_scene, gt_scene, curriculum_distances, device):
        self.params = params
        self.mesh = mesh
        self.macarons = macarons
        self.camera = camera
        self.curriculum_distances = curriculum_distances
        self.surface_scene = surface_scene
        self.covered_scene = covered_scene
        self.proxy_scene = proxy_scene
        self.gt_scene = gt_scene
        self.full_pc = full_pc
        self.full_pc_colors = full_pc_colors
        self.device = device 
        
        self.test_resolution = 0.05        
        self.camera.fov_camera_0 = self.camera.fov_camera

        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=self.camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=self.params.n_alpha,
                                                                                             return_gt_zbuf=True)
        
        for i in range(all_zbuf[-1:].shape[0]):
            part_pc = self.camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                                                                    mask=all_mask[-1:],
                                                                    fov_cameras=self.camera.get_fov_camera_from_RT(
                                                                        R_cam=all_R[-1:],
                                                                        T_cam=all_T[-1:]),
                                                                    gathering_factor=self.params.gathering_factor,
                                                                    fov_range=self.params.sensor_range)
                    
            self.part_pc_features = torch.zeros(len(part_pc), 1, device=self.device)
            self.covered_scene.fill_cells(part_pc, features=self.part_pc_features)
        self.current_coverage = self.gt_scene.scene_coverage(self.covered_scene,
                                                   surface_epsilon=2 * self.test_resolution * self.params.scene_scale_factor)
         
        # self.surface_distance = self.curriculum_distances[self.state_i]

        batch_dict, alpha_dict = create_batch_for_depth_model(params=self.params,
                                                        all_images=all_images,
                                                        all_mask=all_mask,
                                                        all_R=all_R,
                                                        all_T=all_T,
                                                        all_zfar=all_zfar,
                                                        mode='inference',
                                                        device=self.device,
                                                        all_zbuf=all_zbuf)
        with torch.no_grad():
            depth, mask, error_mask, _, _ = apply_depth_model(params=self.params,
                                                            macarons=self.macarons.depth,
                                                            batch_dict=batch_dict,
                                                            alpha_dict=alpha_dict,
                                                            device=self.device,
                                                            use_perfect_depth=self.params.use_perfect_depth)
        for i in range(depth.shape[0]):
            part_pc, part_pc_features = self.camera.compute_partial_point_cloud(depth=depth[i:i+1],
                                                                                images=batch_dict["images"][i:i+1],
                                                                                mask=(mask*error_mask)[i:i+1],
                                                                                fov_cameras=self.camera.get_fov_camera_from_RT(
                                                                                    R_cam=batch_dict['R'][i:i+1],
                                                                                    T_cam=batch_dict['T'][i:i+1]),
                                                                                gathering_factor=self.params.gathering_factor,
                                                                                fov_range=self.params.sensor_range)
            self.surface_scene.fill_cells(part_pc, features=part_pc_features)

            self.full_pc = torch.vstack((self.full_pc, part_pc))
            self.full_pc_colors = torch.vstack((self.full_pc_colors, part_pc_features))
        
        fov_proxy_points, fov_proxy_mask = self.camera.get_points_in_fov(self.proxy_scene.proxy_points,
                                                                        return_mask=True,
                                                                        fov_camera=None,
                                                                        fov_range=self.params.sensor_range)
        fov_proxy_indices = self.proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
        self.proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1,1))

        sgn_dists = self.camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                                depth_maps=depth,
                                                                mask=mask,
                                                                fov_camera=None)
        self.proxy_scene.update_proxy_view_states(self.camera,
                                                  fov_proxy_mask,
                                                  signed_distances=sgn_dists,
                                                  distance_to_surface=None,
                                                  X_cam=None)
        
        self.proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=self.params.carving_tolerance)

        # Update the out-of-field status for proxy points inside camera field of view
        self.proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

        with torch.no_grad():
            self.X_world, self.view_harmonics, self.occ_probs = compute_scene_occupancy_probability_field(self.params, self.macarons.scone,
                                                                                           self.camera,
                                                                                           self.surface_scene, self.proxy_scene,
                                                                                           self.device)     

    def get_current_coverage(self):
        if self.current_coverage[0] == 0.:
            return 0.
        else:
            return self.current_coverage[0].item()
        
    def get_possible_moves(self):
        self.neighbor_indices = self.camera.get_neighboring_poses()
        self.valid_neighbors = self.camera.get_valid_neighbors(neighbor_indices=self.neighbor_indices, mesh=self.mesh)

        self.neighbor_pose_coverage_visibility_list = []
        for neighbor_i in range(len(self.valid_neighbors)):
            neighbor_idx = self.valid_neighbors[neighbor_i]
            neighbor_pose, _ = self.camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, _, fov_neighbor = self.camera.get_camera_parameters_from_pose(neighbor_pose)
            with torch.no_grad():
                _, _, fov_proxy_volume, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                    params=self.params,
                    macarons=self.macarons.scone,
                    proxy_scene=self.proxy_scene,
                    surface_scene=self.surface_scene,
                    X_world=self.X_world,
                    proxy_view_harmonics=self.view_harmonics,
                    occ_probs=self.occ_probs,
                    camera=self.camera,
                    X_cam_world=X_neighbor,
                    fov_camera=fov_neighbor
                )
            # if X_neighbor in self.camera.X_cam_history:
            #     coverage_gain /= 2
            if coverage_gain.shape[0] > 0:
                # pose_distance = calculate_pose_distance(self.camera.X_cam, X_neighbor)
                # efficiency_value = coverage_gain.item() / pose_distance
                if not line_segment_intersects_point_cloud_region(self.gt_scene, self.camera.X_cam, X_neighbor, self.device):
                    self.neighbor_pose_coverage_visibility_list.append((neighbor_idx, coverage_gain, visibility_gains, fov_proxy_volume))
 
        sorted_list = sorted(self.neighbor_pose_coverage_visibility_list, key=lambda item: item[1].item(), reverse=True)

        first_values = [item[0] for item in sorted_list]
        
        # return sorted_coverage_list
        # select top 5 nodes
        if len(first_values) == 0:
            print("God, there is no possible move in the pool")
        elif len(first_values) <= 5:
            return first_values
        else:
            top_five_idx = first_values[:5]
            return top_five_idx

    #-------------Now we will make a fake move---------------------
    def make_a_imaginary_move(self, move_idx):
        interpolation_step = 1
        for _ in range(self.camera.n_interpolation_steps):
            self.camera.update_camera(move_idx, interpolation_step=interpolation_step)
            interpolation_step += 1
        with torch.no_grad():
            self.X_world, self.view_harmonics, self.occ_probs = compute_scene_occupancy_probability_field(self.params, self.macarons.scone,
                                                                                           self.camera,
                                                                                           self.surface_scene, self.proxy_scene,
                                                                                           self.device)
        
        for item in self.neighbor_pose_coverage_visibility_list:
            if torch.equal(item[0], move_idx):
                return item

    #------------Now we will make a real move to next pose-------------
    def make_a_real_move(self, move_idx):
        interpolation_step = 1
        for i in range(self.camera.n_interpolation_steps):
            self.camera.update_camera(move_idx, interpolation_step=interpolation_step)
            self.camera.capture_image(self.mesh)
            interpolation_step += 1

        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
            camera=self.camera,
            n_frames=self.params.n_interpolation_steps,
            n_alpha=self.params.n_alpha_for_supervision,
            return_gt_zbuf=True)

        batch_dict, alpha_dict = create_batch_for_depth_model(params=self.params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='supervision', device=self.device, all_zbuf=all_zbuf)

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
                depth_i, mask_i, error_mask_i, _, _ = apply_depth_model(params=self.params, macarons=self.macarons.depth,
                                                                        batch_dict=batch_dict_i,
                                                                        alpha_dict=alpha_dict_i,
                                                                        device=self.device,
                                                                        compute_loss=False,
                                                                        use_perfect_depth=self.params.use_perfect_depth)

            depth.append(depth_i)
            mask.append(mask_i)
            error_mask.append(error_mask_i)
        depth = torch.cat(depth, dim=0)
        mask = torch.cat(mask, dim=0)
        error_mask = torch.cat(error_mask, dim=0)

        all_part_pc = []
        all_part_pc_features = []
        all_fov_proxy_points = torch.zeros(0, 3, device=self.device)
        general_fov_proxy_mask = torch.zeros(self.params.n_proxy_points, device=self.device).bool()
        all_fov_proxy_mask = []
        all_sgn_dists = []
        all_X_cam = []
        all_fov_camera = []

        for i in range(depth.shape[0]):
            fov_frame = self.camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
            all_X_cam.append(fov_frame.get_camera_center())
            all_fov_camera.append(fov_frame)

            # TO CHANGE: filter points based on SSIM value!
            part_pc, part_pc_features = self.camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         images=batch_dict['images'][i:i+1],
                                                         mask=(mask * error_mask)[i:i + 1].bool(),
                                                         fov_cameras=fov_frame,
                                                         gathering_factor=self.params.gathering_factor,
                                                         fov_range=self.params.sensor_range)

            # Surface points to fill surface scene
            all_part_pc.append(part_pc)
            all_part_pc_features.append(part_pc_features)

            # Get Proxy Points in current FoV
            fov_proxy_points, fov_proxy_mask = self.camera.get_points_in_fov(self.proxy_scene.proxy_points, return_mask=True,
                                                                        fov_camera=fov_frame, fov_range=self.params.sensor_range)
            all_fov_proxy_points = torch.vstack((all_fov_proxy_points, fov_proxy_points))
            all_fov_proxy_mask.append(fov_proxy_mask)
            general_fov_proxy_mask = general_fov_proxy_mask + fov_proxy_mask

            # Computing signed distance of proxy points in fov
            sgn_dists = self.camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth[i:i + 1],
                                                                 mask=mask[i:i + 1].bool(), fov_camera=fov_frame
                                                                 ).view(-1, 1)
            all_sgn_dists.append(sgn_dists)

            # Computing mask for proxy points close to the surface.
            # We will use this for occupancy probability supervision.
            # close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < self.surface_distance)

        # ----------Update Scenes to finalize supervision signal and prepare next iteration-----------------------------

        # 1. Surface scene
        # Fill surface scene
        # We give a visibility=1 to points that were visible in frame t, and 0 to others
        complete_part_pc = torch.vstack(all_part_pc)
        complete_part_pc_features = torch.vstack(all_part_pc_features)
        # complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
        # complete_part_pc_features[:len(all_part_pc[0])] = 1.
        self.surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

        self.full_pc = torch.vstack((self.full_pc, complete_part_pc))
        self.full_pc_colors = torch.vstack((self.full_pc_colors, complete_part_pc_features))

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
        general_fov_proxy_indices = self.proxy_scene.get_proxy_indices_from_mask(general_fov_proxy_mask)
        self.proxy_scene.fill_cells(self.proxy_scene.proxy_points[general_fov_proxy_mask],
                               features=general_fov_proxy_indices.view(-1, 1))

        for i in range(depth.shape[0]):
            # Updating view_state vectors
            self.proxy_scene.update_proxy_view_states(self.camera, all_fov_proxy_mask[i],
                                                 signed_distances=all_sgn_dists[i],
                                                 distance_to_surface=None,
                                                 X_cam=all_X_cam[i])  # distance_to_surface TO CHANGE!

            # Update the supervision occupancy for proxy points using the signed distance
            self.proxy_scene.update_proxy_supervision_occ(all_fov_proxy_mask[i], all_sgn_dists[i],
                                                     tol=self.params.carving_tolerance)

        # Update the out-of-field status for proxy points inside camera field of view
        self.proxy_scene.update_proxy_out_of_field(general_fov_proxy_mask)
        
        return ExploreState(self.mesh, self.params, self.macarons, self.camera, self.full_pc, self.full_pc_colors, self.surface_scene,
                 self.covered_scene, self.proxy_scene, self.gt_scene, self.curriculum_distances, self.device)
    
    def clone(self):
        return copy.deepcopy(self)
    
    def is_terminal(self):
        return False

class CameraState:
    def __init__(self, camera, params, macarons, surface_scene, proxy_scene, gt_scene, mesh, device):
        self.camera = camera
        self.params = params
        self.macarons = macarons
        self.surface_scene = surface_scene
        self.proxy_scene = proxy_scene
        self.gt_scene = gt_scene
        self.device = device
        self.mesh = mesh
        with torch.no_grad():
            self.X_world, self.view_harmonics, self.occ_probs = compute_scene_occupancy_probability_field(self.params, self.macarons.scone,
                                                                                           self.camera,
                                                                                           self.surface_scene, self.proxy_scene,
                                                                                           self.device) 

    def get_possible_moves(self):
        self.neighbor_indices = self.camera.get_neighboring_poses()
        self.valid_neighbors = self.camera.get_valid_neighbors(neighbor_indices=self.neighbor_indices, mesh=self.mesh)

        self.neighbor_pose_coverage_visibility_list = []
        for neighbor_i in range(len(self.valid_neighbors)):
            neighbor_idx = self.valid_neighbors[neighbor_i]
            neighbor_pose, _ = self.camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, _, fov_neighbor = self.camera.get_camera_parameters_from_pose(neighbor_pose)
            with torch.no_grad():
                sample_idx, fov_occ_probs, fov_X_world, fov_X_mask, visibility_gains, coverage_gain = mcts_predict_coverage_gain_for_single_camera(
                    params=self.params,
                    macarons=self.macarons.scone,
                    proxy_scene=self.proxy_scene,
                    surface_scene=self.surface_scene,
                    X_world=self.X_world,
                    proxy_view_harmonics=self.view_harmonics,
                    occ_probs=self.occ_probs,
                    camera=self.camera,
                    X_cam_world=X_neighbor,
                    fov_camera=fov_neighbor
                )
            # if X_neighbor in self.camera.X_cam_history:
            #     coverage_gain /= 2
            if coverage_gain.shape[0] > 0:
                # pose_distance = calculate_pose_distance(self.camera.X_cam, X_neighbor)
                # efficiency_value = coverage_gain.item() / pose_distance
                if not line_segment_intersects_point_cloud_region(self.gt_scene, self.camera.X_cam, X_neighbor, self.device):
                    self.neighbor_pose_coverage_visibility_list.append((neighbor_idx, coverage_gain, visibility_gains, fov_occ_probs, fov_X_mask, sample_idx, fov_X_world))
 
        sorted_list = sorted(self.neighbor_pose_coverage_visibility_list, key=lambda item: item[1].item(), reverse=True)

        # get sorted list and collect first one value and store it.
        first_values = [item[0] for item in sorted_list]
        
        # return sorted_coverage_list-----5 nodes
        if len(first_values) == 0:
            print("God, there is no possible move in the pool")
        elif len(first_values) <= 5:
            return first_values
        else:
            top_five_idx = first_values[:5]
            return top_five_idx

    def clone(self):
        cloned_state = self.__class__(
            camera=copy.deepcopy(self.camera),  # create a deep copy of camera
            params=self.params,  # directly reference the same object
            macarons=self.macarons, 
            surface_scene=self.surface_scene,
            proxy_scene=self.proxy_scene,
            gt_scene=self.gt_scene,
            device=self.device,
            mesh=self.mesh
        )
        cloned_state.X_world = self.X_world
        cloned_state.view_harmonics = copy.deepcopy(self.view_harmonics)
        cloned_state.occ_probs = copy.deepcopy(self.occ_probs)
        cloned_state.neighbor_pose_coverage_visibility_list = self.neighbor_pose_coverage_visibility_list
        return cloned_state
    
    def is_terminal(self):
        return False

    def make_a_imaginary_move(self, move_idx):
        interpolation_step = 1
        for _ in range(self.camera.n_interpolation_steps):
            self.camera.update_camera(move_idx, interpolation_step=interpolation_step)
            interpolation_step += 1
        with torch.no_grad():
            self.X_world, self.view_harmonics, self.occ_probs = compute_scene_occupancy_probability_field(self.params, self.macarons.scone,
                                                                                           self.camera,
                                                                                           self.surface_scene, self.proxy_scene,
                                                                                           self.device)
        
        for item in self.neighbor_pose_coverage_visibility_list:
            if torch.equal(item[0], move_idx):
                return item

    
class Node:
    def __init__(self, state, parent=None, move_index=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.results = 0.0
        self.move_index = move_index
    
    def add_child(self, state, move_index):
        child = Node(state, parent=self, move_index=move_index)
        self.children.append(child)
    
    def select_child(self):
        # Use the Upper Confidence Bound for Trees (UCT) formula to select a child node
        exploration_factor = 1.4
        if self.visits == 0:
            return random.choice(self.children)
        return max(self.children, key=lambda child: child.results / child.visits + exploration_factor * math.sqrt(math.log(self.visits) / child.visits))
    
    def update(self, result):
        self.visits += 1
        self.results += result

class MCTS:
    def __init__(self, initial_state):
        self.root = Node(initial_state)

    def run_search(self, num_iterations):
        for _ in range(num_iterations):
            node = self.select_node()
            result = self.simulation(node)
            self.backpropogation(node, result)

    def select_node(self):
        node = self.root
        while True:
            # Check if the current node can be expanded.
            unvisited_moves = [move for move in node.state.get_possible_moves() if list(move) not in [list(child.move_index) for child in node.children]]
            if unvisited_moves:
                return self.expand(node, unvisited_moves)
            # If the node cannot be expanded because all children are visited,
            # use UCT to select a child node.
            elif node.children:
                # print(22222222222222)
                node = node.select_child()
            else:
                # If there are no children and no unvisited moves, break the loop.
                break
        return node

    def expand(self, node, unvisited_moves):
        random_move = random.choice(unvisited_moves)
        cloned_state = node.state.clone()
        _ = cloned_state.make_a_imaginary_move(random_move)
        node.add_child(cloned_state, random_move)
        # print("111111111111")
        return node.children[-1]
    
    def simulation(self, node):
        state = CameraState(camera=node.state.camera,
                            params=node.state.params,
                            macarons=node.state.macarons,
                            surface_scene=node.state.surface_scene,
                            proxy_scene=node.state.proxy_scene,
                            gt_scene=node.state.gt_scene,
                            mesh=node.state.mesh,
                            device=node.state.device)
        
        moves = 0
        mcts_visibility_gains_list = []
        # Simulation: Move next 5 poses
        while not state.is_terminal() and moves < 5:
            # print("mvs", moves)
            possible_moves = state.get_possible_moves()
            random_move = random.choice(possible_moves)
            result_list = state.make_a_imaginary_move(random_move)
            mcts_visibility_gains_list.append(result_list)
            moves += 1
        reward = compute_coverage_gain_for_path(mcts_visibility_gains_list)
        
        return reward
    
    def backpropogation(self, node, result):
        while node:
            node.update(result)
            node = node.parent

    def get_best_move(self):
        best_child = max(self.root.children, key=lambda child: child.visits)
        return best_child.move_index

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

def plot_scene_and_tragectory_clear(scene_name, scene, camera, device, results_dir):
    X_scene, c_scene = scene.return_entire_pt_cloud(return_features=True)

    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)
    X_cam = camera.X_cam_history + 0.
    print("============", len(X_cam))
    c_cam = torch.zeros_like(X_cam)
    # Create a tensor for camera positions features and color it with a gradient of blue to black
    c_cam[..., 0] += torch.linspace(0.8, 0.0, len(X_cam), device=device)  # Red channel
    c_cam[..., 1] += torch.linspace(1.0, 0.4, len(X_cam), device=device)  # Green channel
    c_cam[..., 2] += torch.linspace(0.8, 0.0, len(X_cam), device=device)  # Blue channel
    # Decrease blue channel over time original:1

    # Create additional points between camera positions to simulate lines
    X_cam_lines = []
    c_cam_lines = []
    for i in range(len(X_cam)-1):
        # Interpolate between two camera points for each coordinate separately
        X_line_x = torch.linspace(X_cam[i][0], X_cam[i+1][0], steps=20, device=device)
        X_line_y = torch.linspace(X_cam[i][1], X_cam[i+1][1], steps=20, device=device)
        X_line_z = torch.linspace(X_cam[i][2], X_cam[i+1][2], steps=20, device=device)
        X_line = torch.stack((X_line_x, X_line_y, X_line_z), dim=-1)

        # Interpolate between the corresponding colors for each channel separately
        c_line_r = torch.linspace(c_cam[i][0], c_cam[i+1][0], steps=20, device=device)  # Switch the order for decreasing red
        c_line_g = torch.linspace(c_cam[i+1][1], c_cam[i][1], steps=20, device=device)
        c_line_b = torch.linspace(c_cam[i+1][2], c_cam[i][2], steps=20, device=device)

        c_line = torch.stack((c_line_r, c_line_g, c_line_b), dim=-1)

        X_cam_lines.append(X_line)
        c_cam_lines.append(c_line)

    # Concatenate the original points with the interpolated points
    X_cam = torch.cat(X_cam_lines, dim=0)
    c_cam = torch.cat(c_cam_lines, dim=0)


    X_perm = torch.randperm(len(X))
    X = torch.cat((X_cam.to(device), X[X_perm][:20000 - len(X_cam)].to(device)), dim=0)
    c = torch.cat((c_cam.to(device), c[X_perm][:20000 - len(c_cam)].to(device)), dim=0)
    X = torch.cat((X, X_scene), dim=0)
    c = torch.cat((c, c_scene), dim=0)

    X_cam_max_y = torch.max(X_cam[:, 1])
    y_threshold = X_cam_max_y + 0.0001
    y_filtered_mask = X[:, 1] <= y_threshold
    # X_cam_max_x = torch.min(X_cam[:, 2])
    # x_threshold = X_cam_max_x - 0.0001
    # x_filtered_mask = X[:, 2] >= x_threshold

    X_filtered = X[y_filtered_mask]
    c_filtered = c[y_filtered_mask]

    dict_to_save = {'points': X_scene.cpu().numpy().tolist(), 'colors': c_scene.cpu().numpy().tolist()}
    reconstructed_json_name = os.path.join(results_dir, "rl_clear_gt_" + scene_name + '.json')

    print(f"Number of surface points to save: {len(dict_to_save['points'])}.")
    print(f"Path to the json file to save: {reconstructed_json_name}")

    with open(reconstructed_json_name, 'w') as outfile:
        json.dump(dict_to_save, outfile)
    
    return plot_point_cloud(X_filtered, c_filtered, name="3D Scene", max_points=150000, point_size=2,
                            width=800, height=600)


def plot_scene_and_tragectory(scene_name, scene, camera, device, results_dir):
    X_scene, c_scene = scene.return_entire_pt_cloud(return_features=True)

    X = torch.zeros(0, 3).to(device)
    c = torch.zeros(0, 3).to(device)
    X_cam = camera.X_cam_history + 0.
    # Create a tensor for camera positions features and color it with a gradient of blue to black
    c_cam = torch.zeros_like(X_cam)
    # Blue channel - starts at 1.0 (for light blue) and decreases to 0.8 (for medium blue)
    # Blue channel - starts at 1.0 and decreases to 0.9
    # c_cam[..., 0] += torch.linspace(1.0, 0.5, len(X_cam), device=device)  # Red channel rrt
    # c_cam[..., 1] += torch.linspace(0.5, 0.0, len(X_cam), device=device)  # Green channel
    # c_cam[..., 2] += torch.linspace(1.0, 1.0, len(X_cam), device=device)  # Blue channel 
    # c_cam[..., 1] += torch.linspace(1.0, 0.5, len(X_cam), device=device)   #nbv  # Blue channel - starts at 0.5 (for light green) and decreases to 0 (for deep green)    
    # c_cam[..., 2] += torch.linspace(0.5, 0.0, len(X_cam), device=device)
    # c_cam[..., 0] += torch.linspace(1.0, 0.0, len(X_cam), device=device) # dij
    # c_cam[..., 2] += torch.linspace(1.0, 0.0, len(X_cam), device=device) #samp
    # Initial color as light yellow pro
    # c_cam[..., 0] += 1.0  # Red channel always 1.0 for yellow
    # c_cam[..., 1] += torch.linspace(1.0, 0.8, len(X_cam), device=device)  # Green channel - starts at 1.0 and decreases to 0.0
    # c_cam[..., 2] += torch.linspace(0.6, 0.0, len(X_cam), device=device)  # Blue channel - starts at 0.6 (for light yellow) and decreases to 0.0 (for black)

    # Blue channel - starts at 1.0 and decreases to 0.9 
    # c_cam[..., 2] += torch.linspace(1.0, 0.9, len(X_cam), device=device) # Red and Green channels - starts at 0.5 and decreases to 0.2 
    # c_cam[..., 0] += torch.linspace(0.5, 0.2, len(X_cam), device=device) 
    # c_cam[..., 1] += torch.linspace(0.5, 0.2, len(X_cam), device=device)

    c_cam[..., 0] += torch.linspace(0.8, 0.0, len(X_cam), device=device)  # Red channel
    c_cam[..., 1] += torch.linspace(1.0, 0.4, len(X_cam), device=device)  # Green channel
    c_cam[..., 2] += torch.linspace(0.8, 0.0, len(X_cam), device=device)  # Blue channel

    # Create additional points between camera positions to simulate lines
    X_cam_lines = []
    c_cam_lines = []
    for i in range(len(X_cam)-1):
        # Interpolate between two camera points for each coordinate separately
        X_line_x = torch.linspace(X_cam[i][0], X_cam[i+1][0], steps=20, device=device)
        X_line_y = torch.linspace(X_cam[i][1], X_cam[i+1][1], steps=20, device=device)
        X_line_z = torch.linspace(X_cam[i][2], X_cam[i+1][2], steps=20, device=device)
        X_line = torch.stack((X_line_x, X_line_y, X_line_z), dim=-1)

        # Interpolate between the corresponding colors for each channel separately
        c_line_r = torch.linspace(c_cam[i][0], c_cam[i+1][0], steps=20, device=device)  # Switch the order for decreasing red
        c_line_g = torch.linspace(c_cam[i+1][1], c_cam[i][1], steps=20, device=device)
        c_line_b = torch.linspace(c_cam[i+1][2], c_cam[i][2], steps=20, device=device)

        c_line = torch.stack((c_line_r, c_line_g, c_line_b), dim=-1)

        X_cam_lines.append(X_line)
        c_cam_lines.append(c_line)

    # Concatenate the original points with the interpolated points
    X_cam = torch.cat(X_cam_lines, dim=0)
    c_cam = torch.cat(c_cam_lines, dim=0)


    X_perm = torch.randperm(len(X))
    X = torch.cat((X_cam.to(device), X[X_perm][:20000 - len(X_cam)].to(device)), dim=0)
    c = torch.cat((c_cam.to(device), c[X_perm][:20000 - len(c_cam)].to(device)), dim=0)
    X = torch.cat((X, X_scene), dim=0)
    c = torch.cat((c, c_scene), dim=0)

    dict_to_save = {'points': X.cpu().numpy().tolist(), 'colors': c.cpu().numpy().tolist()}
    reconstructed_json_name = os.path.join(results_dir, "gt_pro_" + scene_name + '.json')

    print(f"Number of surface points to save: {len(dict_to_save['points'])}.")
    print(f"Path to the json file to save: {reconstructed_json_name}")

    # with open(reconstructed_json_name, 'w') as outfile:
    #     json.dump(dict_to_save, outfile)
    return plot_point_cloud(X, c, name="3D Scene", max_points=150000, point_size=2,
                            width=800, height=600)

def plot_pc_and_tragectory(X_scene, c_scene, camera, device):
    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)
    X_cam = camera.X_cam_history + 0.
    c_cam = torch.zeros_like(X_cam)
    c_cam[..., 0] += torch.linspace(1.0, 0.5, len(X_cam), device=device)  # Red channel
    c_cam[..., 1] += torch.linspace(0.5, 0.0, len(X_cam), device=device)  # Green channel
    c_cam[..., 2] += torch.linspace(1.0, 1.0, len(X_cam), device=device)  # Blue channel

    X_perm = torch.randperm(len(X))
    X = torch.cat((X_cam.to(device), X[X_perm][:20000 - len(X_cam)].to(device)), dim=0)
    c = torch.cat((c_cam.to(device), c[X_perm][:20000 - len(c_cam)].to(device)), dim=0)
    X = torch.cat((X, X_scene), dim=0)
    c = torch.cat((c, c_scene), dim=0)
    return plot_point_cloud(X, c, name="3D Scene", max_points=100000, point_size=2,
                            width=800, height=600)

def plot_scene_and_tragectory_and_constructed_pt(scene_name, params, gt_scene, proxy_scene, surface_scene, macarons, camera, i_th_scene, memory, device, results_dir):
    fig_gt_scene = plot_scene_and_tragectory(scene_name=scene_name, scene=gt_scene, camera=camera, device=device, results_dir=results_dir)
    fig_gt_scene.show()
    fig_gt_scene_clear = plot_scene_and_tragectory_clear(scene_name=scene_name, scene=gt_scene, camera=camera, device=device, results_dir=results_dir)
    fig_gt_scene_clear.show()
    
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
    depths_memory_path = memory.get_trajectory_depths_path(memory.scene_memory_paths[i_th_scene], 0)
    full_pc, full_pc_colors, full_pc_idx = recompute_mapping(params=params, macarons=macarons, camera=camera,
                                                proxy_scene=proxy_scene, surface_scene=surface_scene, device=device, is_master=True,
                                                save_depths=True,
                                                save_depth_every_n_frame=1,
                                                depths_memory_path=depths_memory_path,
                                                return_colors=True)
    
    filtered_n_inside_fov = torch.zeros(len(full_pc), device=device)
    filtered_n_behind_depth = torch.zeros(len(full_pc), device=device)

    tol = params.carving_tolerance
    score_th = params.score_threshold

    for i_frame in range(0, len(os.listdir(depths_memory_path))):
        depth_dict = torch.load(os.path.join(depths_memory_path, str(i_frame) + '.pt'), map_location=device)
        
        depth = depth_dict['depth']
        mask = depth_dict['mask']
        R = depth_dict['R']
        T = depth_dict['T']
        fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        
        _, fov_mask = camera.get_points_in_fov(pts=full_pc, return_mask=True, 
                                                fov_camera=fov_camera, fov_range=params.sensor_range)
        
        fov_pc = full_pc[fov_mask]
        sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_pc, depth_maps=depth, 
                                                                mask=mask.bool(), fov_camera=fov_camera)
        
        filtered_n_inside_fov[fov_mask] += 1
        filtered_n_behind_depth[fov_mask] += sgn_dists.view(-1) >= -tol
    filter_mask = (filtered_n_behind_depth / filtered_n_inside_fov) > score_th

    filtered_pc = 0. + full_pc[filter_mask]
    filtered_colors = 0. + full_pc_colors[filter_mask]
    filtered_idx = 0. + full_pc_idx[filter_mask]

    print(f"The filtered reconstructed surface point cloud contains {filtered_pc.shape[0]} points in total.")

    # Store pc
    dict_to_save = {'points': filtered_pc.cpu().numpy().tolist(), 'colors': filtered_colors.cpu().numpy().tolist(), "idx": filtered_idx.cpu().numpy().tolist()}
    reconstructed_json_name = os.path.join(results_dir, "rl_recons_" + scene_name + '.json')

    print(f"Number of surface points to save: {len(dict_to_save['points'])}.")
    print(f"Path to the json file to save: {reconstructed_json_name}")

    # with open(reconstructed_json_name, 'w') as outfile:
    #     json.dump(dict_to_save, outfile)
    fild = plot_point_cloud(filtered_pc, filtered_colors, name='Filtered reconstructed surface points', 
                    point_size=2, max_points=150000, width=800, height=600, cmap='rgb')
    
    print("111111", len(full_pc))
    print("111111111", len(filtered_pc))
    fild.show()