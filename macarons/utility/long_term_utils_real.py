import threading
import torch
import copy
import time
import math
import logging
import random
import concurrent.futures
from macarons.utility.render_utils import plot_point_cloud
import matplotlib.pyplot as plt
from macarons.utility.macarons_utils import *
from macarons.utility.utils import count_parameters
from macarons.testers.scene import *
from macarons.utility.render_utils import plot_point_cloud, plot_graph
from torch.nn.functional import pairwise_distance

# from macarons.utility.long_term_utils_real import *
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(threadName)-10s) %(message)s')


#====================MCTS implement========================

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

        # if self.state_i > 0 and self.state_i % self.params.recompute_surface_every_n_loop == 0:
        #    print("Recomputing surface...")
        #    fill_surface_scene(self.surface_scene, self.full_pc,
        #                        random_sampling_max_size=self.params.n_gt_surface_points,
        #                        min_n_points_per_cell_fill=3,
        #                        progressive_fill=self.params.progressive_fill,
        #                        max_n_points_per_fill=self.params.max_points_per_progressive_fill,
        #                        full_pc_colors=self.full_pc_colors)

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

        neighbor_pose_coverage_distance_dict = {}
        for neighbor_i in range(len(self.valid_neighbors)):
            neighbor_idx = self.valid_neighbors[neighbor_i]
            neighbor_pose, _ = self.camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, _, fov_neighbor = self.camera.get_camera_parameters_from_pose(neighbor_pose)
            with torch.no_grad():
                _, _, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
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
                pose_distance = calculate_pose_distance(self.camera.X_cam, X_neighbor)
                efficiency_value = coverage_gain.item() / pose_distance
                if not line_segment_intersects_point_cloud_region(self.gt_scene, self.camera.X_cam, X_neighbor, self.device):
                    neighbor_pose_coverage_distance_dict[neighbor_idx] = (coverage_gain, pose_distance, efficiency_value)
 
        sorted_coverage_list = sorted(neighbor_pose_coverage_distance_dict.keys(), key=lambda k: neighbor_pose_coverage_distance_dict[k][0], reverse=True)
        return sorted_coverage_list
        # if len(sorted_coverage_list) <= 5:
        #     return sorted_coverage_list
        # else:
        #     top_five_idx = sorted_coverage_list[:5]
        #     return top_five_idx
    
    def make_move(self, move_idx):
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

        close_fov_proxy_mask = torch.zeros(self.params.n_proxy_points, device=self.device).bool()

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
    
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        # logging.info(f'Node created with state={state}, action={action}')

class MCTS:
    def __init__(self, iterations=100, exploration_weight=2):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        # Adding a lock for synchronization
        self.update_lock = threading.Lock()

    def run_simulation(self, root):
        # logging.info('Starting simulation')
        node = self.tree_policy(root)
        reward = self.default_policy(node.state)
        self.backpropagate(node, reward)
        # logging.info('Ending simulation')

    def search(self, initial_state):
        root = Node(initial_state)

        # Create a thread pool for parallel simulations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the simulation tasks and wait for them to complete
            futures = [executor.submit(self.run_simulation, root) for _ in range(self.iterations)]
            concurrent.futures.wait(futures)
        print("num of root's children===", len(root.children))
        return self.best_child(root, 0).state

    def tree_policy(self, node):
        while not node.state.is_terminal():
            if not self.is_fully_expanded(node):
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def is_fully_expanded(self, node):
        return len(node.children) == len(node.state.get_possible_moves())


    def expand(self, node):
        possible_moves = node.state.get_possible_moves()
        print("possbile_moves----", len(possible_moves))
        for move in possible_moves:
            new_state = node.state.clone()
            new_state.make_move(move)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
            print("new_children: ===", len(node.children))
            return new_node

        return node


    def best_child(self, node, exploration_weight=None):
        if exploration_weight is None:
            exploration_weight = self.exploration_weight

        # Create a list to store child nodes along with their UCB values
        children_with_ucb = []

        # Lock when accessing children
        with self.update_lock:
            for child in node.children:
                if child.visits == 0:
                    ucb = float("inf")
                else:
                    ucb = (child.value / child.visits) + exploration_weight * math.sqrt(
                    (2 * math.log(node.visits)) / child.visits)
                children_with_ucb.append((child, ucb))

            children_with_ucb.sort(key=lambda x: x[1], reverse=True)
            print("child----", len(children_with_ucb))
            best_child = None
            for child, ucb in children_with_ucb:
                if (child.state.camera.X_cam[:, None, :] == node.state.camera.X_cam_history).all(dim=-1).any(dim=-1):
                    pass
                else:
                    best_child = child
                    break
            if best_child is None:
                print("best child is none, we have to go back")
                return children_with_ucb[0][0]
        return best_child

    def default_policy(self, state):
        simulation_state = state.clone()
        original_coverage = simulation_state.get_current_coverage()
        moves = 0
        while not simulation_state.is_terminal() and moves < 5:
            possible_moves = simulation_state.get_possible_moves()
            move = random.choice(possible_moves)
            simulation_state = simulation_state.make_move(move)
            moves += 1

        # _, counts = torch.unique(simulation_state.camera.X_cam_history, return_inverse=True)
        # counts = torch.bincount(counts)
        
        # duplicate_counts = counts[counts > 1]
        # number_of_duplicates = duplicate_counts.sum().item() - len(duplicate_counts)
        
        return simulation_state.get_current_coverage() - original_coverage 

    def backpropagate(self, node, reward):
        while node is not None:
            with self.update_lock:
                node.visits += 1
                node.value += reward
            node = node.parent

# class Node:
#     def __init__(self, state, parent=None):
#         self.state = state
#         self.parent = parent
#         self.children = []
#         self.visits = 0
#         self.value = 0
#         # logging.info(f'Node created with state={state}, action={action}')

# class MCTS:
#     def __init__(self, iterations=100, exploration_weight=1):
#         self.iterations = iterations
#         self.exploration_weight = exploration_weight
#         # Adding a lock for synchronization
#         # self.update_lock = threading.Lock() 

#     def run_simulation(self, root):
#         # logging.info('Starting simulation')
#         node = self.tree_policy(root)
#         reward = self.default_policy(node.state)
#         self.backpropagate(node, reward)
#         # logging.info('Ending simulation')

#     def search(self, initial_state):
#         root = Node(initial_state)

#         # # Create a thread pool for parallel simulations
#         # with concurrent.futures.ThreadPoolExecutor() as executor:
#         #     # Submit the simulation tasks and wait for them to complete
#         #     futures = [executor.submit(self.run_simulation, root) for _ in range(self.iterations)]
#         #     concurrent.futures.wait(futures)
#         for _ in range(self.iterations):
#             self.run_simulation(root)

#         return self.best_child(root, 0).state

#     def tree_policy(self, node):
#         while not node.state.is_terminal():
#             if not self.is_fully_expanded(node):
#                 return self.expand(node)
#             else:
#                 node = self.best_child(node)
#         return node

#     def is_fully_expanded(self, node):
#         return len(node.children) == len(node.state.get_possible_moves())


#     # def expand(self, node):
#     #     with self.update_lock:
#     #         tried_moves = [child.action for child in node.children]
#     #         possible_moves = node.state.get_possible_moves()
#     #         # print(f"tried_moves: {tried_moves}")
#     #         # print(f"possible_moves: {possible_moves}")
#     #         for move in possible_moves:
#     #             if not any(torch.all(move == tried_move) for tried_move in tried_moves):
#     #                 new_state = node.state.clone()
#     #                 new_state.make_move(move)
#     #                 new_node = Node(new_state, parent=node, action=move)
#     #                 node.children.append(new_node)
#     #                 return new_node

#     #     return node
#     def expand(self, node):
#         # with self.update_lock:
#         possible_moves = node.state.get_possible_moves()
        
#         if possible_moves:
#             move = possible_moves[0]
#             new_state = node.state.clone()
#             new_state.make_move(move)
#             new_node = Node(new_state, parent=node)
#             node.children.append(new_node)
#             return new_node

#         return node


#     def best_child(self, node, exploration_weight=None):
#         if exploration_weight is None:
#             exploration_weight = self.exploration_weight

#         # Create a list to store child nodes along with their UCB values
#         children_with_ucb = []

#         # Lock when accessing children
#         # with self.update_lock:
#         for child in node.children:
#             if child.visits == 0:
#                 ucb = float("inf")
#             else:
#                 ucb = (child.value / child.visits) + exploration_weight * math.sqrt(
#                 (2 * math.log(node.visits)) / child.visits)
#             children_with_ucb.append((child, ucb))

#         children_with_ucb.sort(key=lambda x: x[1], reverse=True)
#         # print("child----", len(children_with_ucb))
#         best_child = None
#         for child, ucb in children_with_ucb:
#             if (child.state.camera.X_cam[:, None, :] == node.state.camera.X_cam_history).all(dim=-1).any(dim=-1):
#                 # print(child.state.camera.X_cam)
#                 # print(node.state.camera.X_cam_history)
#                 pass
#             else:
#                 best_child = child
#                 break
#         if best_child is None:
#             print("dam")
#             print(children_with_ucb)
#         return best_child


#     def default_policy(self, state):
#         simulation_state = state.clone()
#         original_coverage = simulation_state.get_current_coverage()
#         moves = 0
#         while not simulation_state.is_terminal() and moves < 5:
#             possible_moves = simulation_state.get_possible_moves()
#             # print("moves", moves)
#             move = random.choice(possible_moves)
#             simulation_state = simulation_state.make_move(move)
#             moves += 1

#         # _, counts = torch.unique(simulation_state.camera.X_cam_history, return_inverse=True)
#         # counts = torch.bincount(counts)
        
#         # duplicate_counts = counts[counts > 1]
#         # number_of_duplicates = duplicate_counts.sum().item() - len(duplicate_counts)
        
#         return simulation_state.get_current_coverage() - original_coverage

#     def backpropagate(self, node, reward):
#         while node is not None:
#             # with self.update_lock:
#             node.visits += 1
#             node.value += reward
#             node = node.parent


#====================Useful functions========================

def line_segment_intersects_point_cloud_region(scene, start_point, end_point, device):
    point_cloud = scene.return_entire_pt_cloud(return_features=False)
    line_vector = end_point - start_point

    # Calculate the vectors from points to the line segment start and end points
    point_to_start = point_cloud - start_point
    point_to_end = point_cloud - end_point

    # Calculate the projection vectors from points to the line segment start point
    projection_vector = torch.sum(point_to_start * line_vector, dim=1) / torch.norm(line_vector)**2

    # Calculate the distances from points to the line segment
    distances = torch.zeros(point_cloud.shape[0]).to(device)

    # Create a mask to identify points within the segment range
    within_segment_mask = (projection_vector >= 0) & (projection_vector <= 1)

    # Calculate distances for points within the segment range
    if within_segment_mask.any():
        closest_point = start_point + projection_vector[within_segment_mask].unsqueeze(1) * line_vector
        distances_within_segment = torch.norm(point_cloud[within_segment_mask] - closest_point, dim=1)
        distances[within_segment_mask] = distances_within_segment

    # Calculate distances for points outside the segment range
    distances[~within_segment_mask] = torch.min(torch.norm(point_to_start[~within_segment_mask], dim=1),
                                                torch.norm(point_to_end[~within_segment_mask], dim=1))
    
    #torch.norm(end_point-start_point).item()/20
    if torch.min(distances).item() < 2:
        return True
    return False


def calculate_pose_distance(T1, T2):

    translation_distance = torch.norm(T1 - T2)

    distance = torch.sqrt(translation_distance**2)

    return distance

def plot_scene_and_tragectory(scene, camera, device):
    X_scene, c_scene = scene.return_entire_pt_cloud(return_features=True)

    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)
    X_cam = camera.X_cam_history + 0.
    print("============", len(X_cam))
    c_cam = torch.zeros_like(X_cam)
    c_cam[..., 0] += 1
    c_cam[..., 1] += torch.linspace(0.9, 0.0, len(X_cam), device=device)
    c_cam[..., 2] += torch.linspace(0.9, 0.0, len(X_cam), device=device)

    X_perm = torch.randperm(len(X))
    X = torch.cat((X_cam.to(device), X[X_perm][:20000 - len(X_cam)].to(device)), dim=0)
    c = torch.cat((c_cam.to(device), c[X_perm][:20000 - len(c_cam)].to(device)), dim=0)
    X = torch.cat((X, X_scene), dim=0)
    c = torch.cat((c, c_scene), dim=0)
    return plot_point_cloud(X, c, name="3D Scene", max_points=30000, point_size=2,
                            width=730, height=600)

def plot_pc_and_tragectory(X_scene, c_scene, camera, device):
    X = torch.zeros(0, 3)
    c = torch.zeros(0, 3)
    X_cam = camera.X_cam_history + 0.
    c_cam = torch.zeros_like(X_cam)
    c_cam[..., 0] += 1
    c_cam[..., 1] += torch.linspace(0.9, 0.0, len(X_cam), device=device)
    c_cam[..., 2] += torch.linspace(0.9, 0.0, len(X_cam), device=device)

    X_perm = torch.randperm(len(X))
    X = torch.cat((X_cam.to(device), X[X_perm][:20000 - len(X_cam)].to(device)), dim=0)
    c = torch.cat((c_cam.to(device), c[X_perm][:20000 - len(c_cam)].to(device)), dim=0)
    X = torch.cat((X, X_scene), dim=0)
    c = torch.cat((c, c_scene), dim=0)
    return plot_point_cloud(X, c, name="3D Scene", max_points=30000, point_size=2,
                            width=730, height=600)