import os
import gc
import json
import time
import ast

import matplotlib.pyplot as plt
from macarons.utility.macarons_utils import *
from macarons.testers.scene import *
from next_best_path.utility.long_term_utils import *
from pytorch3d.vis.plotly_vis import plot_scene

from next_best_path.networks.nbp_model import *
from next_best_path.utility.nbp_utils import *
from next_best_path.utility.utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../data")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def compute_nbp_trajectory(params, nbp,
                       camera,
                       gt_scene,
                       mesh,
                       mesh_for_check,
                       n_pieces,
                       y_bins,
                       device,
                       test_resolution=0.05,
                       use_perfect_depth_map=True,
                        ):

    t1 = time.time()
    nbp.eval()

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)

    coverage_evolution = []
    
    pc2img_size = (256, 256) # model input size
    prediction_range = (-40, 40)
    value_map_size = (64, 64) # size of value map: [64, 64, 8]
    
    Dijkstra_path = []
    path_record = 0
    collision_list = []
    passable_list = []
    
    splited_pose_space_idx = camera.generate_new_splited_dict()
    splited_pose_space = generate_key_value_splited_dict(splited_pose_space_idx)
    
    # compute the bounding box for the gt_pc
    gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)
    print("size of gt pc: ", gt_scene_pc.shape)

    idx_history = torch.zeros(0, 5, device=device)
    for pose_i in range(101): # n_poses_in_trajectory == 100

        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        camera.fov_camera_0 = camera.fov_camera
        
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=params.n_alpha,
                                                                                             return_gt_zbuf=True)
        # Compute current coverage
        current_exploration_coverage = calculate_coverage_percentage(gt_scene_pc, full_pc)
        print(current_exploration_coverage)
        coverage_evolution.append(current_exploration_coverage)

        
        batch_dict, alpha_dict = create_batch_for_depth_model(params=params,
                                                              all_images=all_images, all_mask=all_mask,
                                                              all_R=all_R, all_T=all_T, all_zfar=all_zfar,
                                                              mode='inference', device=device,
                                                              all_zbuf=all_zbuf)

        # Depth obtain
        with torch.no_grad():
            depth, mask, error_mask, pose, gt_pose = obtain_depth(params=params,
                                                                batch_dict=batch_dict,
                                                                alpha_dict=alpha_dict,
                                                                device=device,
                                                                use_perfect_depth=params.use_perfect_depth)

        if use_perfect_depth_map:
            depth = all_zbuf[2:3]
            error_mask = mask

        # We project the depth map to point cloud
        for i in range(depth.shape[0]):
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
        # We then convert n pieces of point cloud to n images
        for i in range(n_pieces):
            if len(full_pc_groups[i]) > 0:
                points_2d_batch = transform_points_to_n_pieces(full_pc_groups[i], camera_current_pose, device)
                current_partial_pc_img = map_points_to_n_imgs(points_2d_batch, pc2img_size, prediction_range, device) # size(1, 100, 100)
            else:
                current_partial_pc_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device)
            full_pc_images.append(current_partial_pc_img)

        full_pc_images = torch.cat(full_pc_images, dim=0)
        current_pc_imgs = full_pc_images.unsqueeze(0)

        # previsous camera trajectory projection:
        trajectory_2d = transform_points_to_n_pieces(camera.X_cam_history, camera_current_pose, device)
        previous_trajectory_img = map_points_to_n_imgs(trajectory_2d, pc2img_size, prediction_range, device)
        current_previous_trajectory_img = previous_trajectory_img.unsqueeze(0)

        if pose_i == 0:
            dij_condition = True
        else:
            if path_record + 1 > len(Dijkstra_path):
                dij_condition = True
            else:
                camera_next_pose, _ = camera.get_pose_from_idx(Dijkstra_path[path_record])
                # dij_condition = check_line_no_collision(gt_scene_pc, camera_current_pose[:3], camera_next_pose[:3])
                dij_condition = line_segment_mesh_intersection(camera_current_pose[:3], camera_next_pose[:3], mesh_for_check)
                if dij_condition:
                    collision_list.append([camera.cam_idx[:3].tolist(), Dijkstra_path[path_record][:3].tolist()])
                    collision_list.append([Dijkstra_path[path_record][:3].tolist(), camera.cam_idx[:3].tolist()])
                    collision_list.append(Dijkstra_path[-1][:3].tolist())
    
        if len(idx_history) >= 2:
            pass_1 = idx_history[-1].cpu().numpy().tolist()[:3]
            pass_1 = [int(x) for x in pass_1]
            pass_2 = idx_history[-2].cpu().numpy().tolist()[:3]
            pass_2 = [int(x) for x in pass_2]

            passable_list.append([pass_1, pass_2])
            passable_list.append([pass_2, pass_1])

        #---------------------------------------------------------------------------------------------------------
        # Dijkstra path generation: When we meet a potential collision or in the end of the last Dijkstra path, we regenerate the Dijkstra path
        #---------------------------------------------------------------------------------------------------------
        if dij_condition:
            print("regenerate Dijkstra path")

            Dijkstra_path = []
            path_record = 0
            # value map and obstacle map prediction
            predicted_value_map, predicted_obstacle_map = nbp(torch.cat((current_pc_imgs, current_previous_trajectory_img), dim=1).to(device))
            # Covert the predicted_obstacle_map to binary image
            threshold = 0.13
            predicted_obstacle_map = (predicted_obstacle_map >= threshold).float()

            # Covert current point cloud to a single binary image
            full_pc_trans_points = transform_points_to_n_pieces(full_pc, camera_current_pose, device)
            full_pc_projection = map_points_to_n_imgs(full_pc_trans_points, pc2img_size, prediction_range, device) 
            full_pc_projection = full_pc_projection.unsqueeze(0)
            full_pc_projection[full_pc_projection > 1] = 1

            # Generate point cloud mask for current height layer
            filt_pc_mask =  (full_pc[:, 1] < camera_current_pose[1].item()+0.1) & (full_pc[:, 1] > camera_current_pose[1].item()-0.1)
            filt_pc = full_pc[filt_pc_mask]
            filt_pc_selection = transform_points_to_n_pieces(filt_pc, camera_current_pose, device)
            filt_pc_img = map_points_to_n_imgs(filt_pc_selection, pc2img_size, prediction_range, device) 
            filt_pc_selection_img = filt_pc_img.unsqueeze(0)
            filt_pc_selection_img[filt_pc_selection_img > 0] = 1

            # Fuse real obstacle map and predicted obstacle map
            mask_layout = full_pc_projection > 0
            predicted_obstacle_map[mask_layout] = filt_pc_selection_img[mask_layout]

            # Set previous trajectory positions as passable (value 0)
            mask_trajectory = current_previous_trajectory_img > 0
            predicted_obstacle_map[mask_trajectory] = 0

            # Calculate the best value scores for each candidate positions
            max_gain_map, _ = torch.max(predicted_value_map, dim=1, keepdim=True)
            camera_position_value_list = []
            
            # Get current camera position in img
            camera_points = transform_points_to_n_pieces(camera_current_pose[:3].unsqueeze(0), camera_current_pose, device)
            camera_grid_position = get_point_position_in_the_img(camera_points.squeeze(0), pc2img_size, prediction_range)


            # Iterate through all possible camera positions, calculate best value for each position
            for key, point_3d in splited_pose_space.items():
                # Skip known collision positions
                if ast.literal_eval(key) in collision_list:
                    continue
                
                # Convert 3D position to 2D grid coordinates
                point_2d = transform_points_to_n_pieces(point_3d.unsqueeze(0), camera_current_pose, device) 
                grid_position = get_point_position_in_the_img(point_2d.squeeze(0), value_map_size, prediction_range)

                # Check if position is within valid range
                if 0 <= grid_position[0] < value_map_size[0] and 0 <= grid_position[1] < value_map_size[1]:
                    # Change to locations in image: (H - 1 - y, x)
                    cam_img_position = torch.tensor([grid_position[0], grid_position[1]]).to(device)
                    # Get value of this position from value map
                    value_result = max_gain_map[0, 0, cam_img_position[0], cam_img_position[1]].detach()

                    grid_position_select = get_point_position_in_the_img(point_2d.squeeze(0), pc2img_size, prediction_range)
                    cam_img_position_select = torch.tensor([grid_position_select[0], grid_position_select[1]]).to(device)
                    score_for_select = full_pc_projection[0, 0, cam_img_position_select[0], cam_img_position_select[1]].detach()

                    # Check if this pixel position is valid
                    if check_pixel_values(full_pc_projection, cam_img_position_select):
                        new_position_values = []
                        new_position_values.append(key)
                        new_position_values.append(cam_img_position)
                        # Calculate comprehensive score: gain value - 10 * point cloud density penalty, can remove
                        new_position_values.append(value_result.item() - 10 * score_for_select.item())

                        camera_position_value_list.append(new_position_values)
            # Sort candidate positions by value score in descending order
            camera_position_value_list.sort(key=lambda x: x[-1], reverse=True)
            
            for pose_location in camera_position_value_list:

                path_start_position = camera.cam_idx[:3].tolist()
                path_end_position = ast.literal_eval(pose_location[0])

                Dijkstra_path = generate_Dijkstra_path(splited_pose_space, path_start_position, path_end_position, gt_scene, camera_current_pose, camera, 
                                                        value_map_size, prediction_range, predicted_value_map, device, layout_image=predicted_obstacle_map, layout_size=pc2img_size, collision_list=collision_list, training_flag=False, passable_list=passable_list)
                # Check if generated path is valid and safe
                if Dijkstra_path is not None and Dijkstra_path.nelement() > 0:
                    camera_next_pose, _ = camera.get_pose_from_idx(Dijkstra_path[path_record])
                    if not line_segment_mesh_intersection(camera_current_pose[:3], camera_next_pose[:3], mesh_for_check):
                        break
                    else:
                        collision_list.append([camera.cam_idx[:3].tolist(), Dijkstra_path[path_record][:3].tolist()])
                        collision_list.append([Dijkstra_path[path_record][:3].tolist(), camera.cam_idx[:3].tolist()])
    
        else:
            predicted_value_map, _ = nbp(torch.cat((current_pc_imgs, current_previous_trajectory_img), dim=1).to(device))

        # Handle invalid path situations
        if Dijkstra_path is None:
            # When no valid path exists, randomly select a rotation
            next_idx[-1] = torch.randint(0, 8, (1,))
            Dijkstra_path = []
        else:
            # Get next action from path
            next_idx = Dijkstra_path[path_record]
            matches = torch.all(next_idx == idx_history, dim=1)
            exists = torch.any(matches).item()
            if exists:
                next_idx[-1] = torch.randint(0, 8, (1,))

        # ----------Move to next camera pose----------------------------------------------------------------------------
        # We move to the next pose and capture RGB images.
        idx_history = torch.vstack((idx_history, camera.cam_idx.unsqueeze(0)))
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1
        
        # ----------Depth gain------------------------------------------------------------------------------------
        # Load input RGB image and camera pose
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

        all_part_pc = []
        all_part_pc_features = []
        all_X_cam = []
        all_fov_camera = []


        for i in range(depth.shape[0]):
            fov_frame = camera.get_fov_camera_from_RT(R_cam=batch_dict['R'][i:i + 1], T_cam=batch_dict['T'][i:i + 1])
            all_X_cam.append(fov_frame.get_camera_center())
            all_fov_camera.append(fov_frame)

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

    t2 = time.time()
    print("Time: ", t2-t1)

    
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, full_pc, full_pc_colors


def test_nbp_planning(params_file,
            model_file,
            results_json_file,
            numGPU,
            test_scenes,
            test_resolution=0.05,
            use_perfect_depth_map=False,
            compute_collision=False,
            load_json=False,
            dataset_path=None,
            nbp_weights=None):
    params_path = os.path.join(configs_dir, params_file)
    weights_path = os.path.join(weights_dir, model_file,)
    results_json_path = os.path.join(results_dir, results_json_file)
    nbp_weights = os.path.join(dir_path, nbp_weights)

    params = load_params(params_path)
    params.test_scenes = test_scenes

    results_to_save = {}

    params.jitter_probability = 0. # Probability to apply color jitter on images during training. 
    params.symmetry_probability = 0. # Probability to symmetrize the mesh along any specified axis when loading a mesh of a scene during training. 
    params.anomaly_detection = False
    params.memory_dir_name = "test_memory_" + str(numGPU)

    params.jz = False
    params.ddp = False

    params.numGPU = numGPU
    params.WORLD_SIZE = 1
    params.batch_size = 1
    params.total_batch_size = 1


    if dataset_path is None:
        params.data_path = data_path
    else:
        params.data_path = dataset_path

    device = setup_device(params, None)
    
    nbp = NBP()
    chechp = torch.load(nbp_weights, map_location=device)
    nbp.load_state_dict(chechp['model_state_dict'])
    nbp.to(device)
    # Setup model and dataloader, only get the test dataset
    dataloader, memory = setup_nbp_test(params, weights_path, device) 

    
    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
    
        for i_th_scene in range(len(scene_names)):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_th_scene]
            obj_name = obj_names[i_th_scene]
            settings = all_settings[i_th_scene]
            settings = Settings(settings, device, params.scene_scale_factor)

            print("\nScene name:", scene_name)
            print("-------------------------------------")

            results_to_save[scene_name] = {}

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)

            mirrored_scene = False # (bool) If True, mirrors the coordinates on x-axis
            mirrored_axis = None

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
            
            # split point cloud to n pieces
            n_pieces = 4
            verts = mesh.verts_list()[0]
            min_y, max_y = torch.min(verts, dim=0)[0][1].item() + 0.5, torch.max(verts, dim=0)[0][1].item() - 0.5

            bin_width = (max_y - min_y) / n_pieces
            y_bins = torch.arange(min_y, max_y+bin_width, bin_width, device=device)

            # store a mesh via trimesh for collision check
            mesh_for_check = trimesh.load(mesh_path)
            mesh_for_check.vertices *= params.scene_scale_factor
            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])
            
            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)

            torch.cuda.empty_cache()
            
            for start_cam_idx_i in range(len(settings.camera.start_positions)):
                start_cam_idx = settings.camera.start_positions[start_cam_idx_i]
                print("Start cam index for " + scene_name + ":", start_cam_idx)

                # Setup the Scene and Camera objects
                gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
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
                occupied_pose_data = None
                camera = setup_test_camera(params, mesh, mesh_for_check, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.cam_idx_history)
                print(camera.X_cam_history)
                print(camera.X_cam_history[0], camera.V_cam_history[0])
                
                
                coverage_evolution, X_cam_history, V_cam_history, full_pc, full_pc_colors = compute_nbp_trajectory(params, nbp,
                                                                                      camera,
                                                                                      gt_scene,
                                                                                      mesh,
                                                                                      mesh_for_check,
                                                                                      n_pieces,
                                                                                      y_bins,
                                                                                      device,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      )
                
                
                results_to_save[scene_name][str(start_cam_idx_i)] = {}
                results_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                results_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                results_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()
                
                with open(results_json_path, 'w') as outfile:
                    json.dump(results_to_save, outfile)
                print("Saved data about test losses in", results_json_file)

                #-----------------------

    print("All trajectories computed.")

            


