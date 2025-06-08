import os
import sys
import gc
import json
import time
import random

import matplotlib.pyplot as plt
from macarons.utility.macarons_utils import *
from macarons.utility.utils import count_parameters
from macarons.testers.scene import *
from macarons.utility.render_utils import plot_point_cloud, plot_graph
from torch.nn.functional import pairwise_distance
from macarons.trainers.train_macarons import recompute_mapping
from next_best_path.utility.long_term_utils import *


dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/navi_test")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def compute_random_walk_trajectory(params, macarons,
                       camera,
                       gt_scene, surface_scene, proxy_scene, covered_scene,
                       mesh,
                       device,
                       test_resolution=0.05,
                       use_perfect_depth_map=False,
                       compute_collision=True):

    macarons.eval()
    curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
    curriculum_n_cells = get_curriculum_sampling_cell_number(params)

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)

    gt_scene_pc = gt_scene.return_entire_pt_cloud(return_features=False)

    coverage_evolution = []
    t1 = time.time()
    # time

    for pose_i in range(200): # n_poses_in_trajectory == 100
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill,
                               full_pc_colors=full_pc_colors)

        # ----------Predict visible surface points from RGB images------------------------------------------------------

        # Load input RGB image and camera pose
        # Current pose: R_cam=all_R[-1:], T_cam=all_T[-1:]
        
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=params.n_alpha,
                                                                                             return_gt_zbuf=True)
        # print("size of all_R", all_R.size())
        # print("size of all_T", all_T.size())
        
        # Register GT surface points to compute true coverage for evaluation
        for i in range(all_zbuf[-1:].shape[0]):
            # TO CHANGE: filter points based on SSIM value!
            part_pc = camera.compute_partial_point_cloud(depth=all_zbuf[-1:],
                                                         mask=all_mask[-1:],
                                                         fov_cameras=camera.get_fov_camera_from_RT(
                                                             R_cam=all_R[-1:],
                                                             T_cam=all_T[-1:]),
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Fill surface scene
            part_pc_features = torch.zeros(len(part_pc), 1, device=device)
            covered_scene.fill_cells(part_pc, features=part_pc_features)  # Todo: change for better surface
        # Compute true coverage for evaluation
        current_coverage = gt_scene.scene_coverage(covered_scene,
                                                   surface_epsilon=2 * test_resolution * params.scene_scale_factor)
        if pose_i % 10 == 0:
            print("current coverage:", current_coverage)
        
        if current_coverage[0] == 0.:
            coverage_evolution.append(0.)
        else:
            coverage_evolution.append(current_coverage[0].item())



        # surface_distance = curriculum_distances[pose_i]

        # Format input as batches to feed depth model
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
        # end_t = time.time()
        # print("woruiiiiiiiii---", end_t-t0)
        # occupancy_X = X_world + 0
        # occupancy_sigma = occ_probs + 0
        # ----------Predict Coverage Gain of neighbor camera poses------------------------------------------------------

        # Compute valid neighbor poses
        neighbor_indices = camera.get_neighboring_poses_2d()
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        max_coverage_gain = -1.
        next_idx = valid_neighbors[0]

        neighbor_poses_coverage_distance_dict = {}
        # For each valid neighbor...
        for neighbor_i in range(len(valid_neighbors)):
            neighbor_idx = valid_neighbors[neighbor_i]
            neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx) # return corresponding 5D pose
            # R_cam_pose_i, T_cam_pose_i = get_camera_RT(neighbor_pose[:3].view(1, 3), neighbor_pose[3:].view(1, 2))
            X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
            # We check, if needed, if camera collides
            drop_neighbor = False
            compute_collision = False
            if compute_collision:
                # drop_neighbor = proxy_scene.camera_collides(params, camera, X_neighbor)
                drop_neighbor = proxy_scene.path_collides(params, camera, X_neighbor)

            if not drop_neighbor:
                # ...We predict its coverage gain...
                with torch.no_grad():
                    _, _,fov_proxy_volume, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                        params=params, macarons=macarons.scone,
                        proxy_scene=proxy_scene, surface_scene=surface_scene,
                        X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                        camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)
                # ...And save it with the neighbor index if the estimated coverage is better  coverage_gain: tensor(1,1)
                if coverage_gain.shape[0] > 0:
                    pose_distance = calculate_pose_distance(camera.X_cam, X_neighbor)
                    # pose_distance = calculate_pose_distance(all_R[-1:], camera.X_cam, R_cam_pose_i, T_cam_pose_i)
                    # print("mmmmmmmmmmmmmmmmmmmmmmm")
                    # print("camera.X_cam------", camera.X_cam)
                    # print("all_T[-1]-------", all_T[-1])
                    # print("X_neighbor------", X_neighbor)
                    efficiency_value = coverage_gain.item() / pose_distance
                    # print(V_neighbor)
                    if not line_segment_intersects_point_cloud_region(gt_scene, camera.X_cam, X_neighbor, device):
                        # coverage_gain.zero_()
                        neighbor_poses_coverage_distance_dict[neighbor_idx] = (coverage_gain, pose_distance, efficiency_value)

                        if coverage_gain.shape[0] > 0 and coverage_gain.item() > max_coverage_gain:
                            max_coverage_gain = coverage_gain.item()
                            next_idx = neighbor_idx

        # ==============================================================================================================
        # Move to the neighbor NBV and acquire signal
        # ==============================================================================================================

        # Now that we have estimated the NBV among neighbors, we move toward this new camera pose and save RGB images
        # along the way.

        # ----------Move to next camera pose----------------------------------------------------------------------------
        # We move to the next pose and capture RGB images.
        if random.random() < 0.2:
            next_idx = random.choice(list(neighbor_poses_coverage_distance_dict.keys()))
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1

        # ----------Depth prediction------------------------------------------------------------------------------------
        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(
            camera=camera,
            n_frames=params.n_interpolation_steps,
            n_alpha=params.n_alpha_for_supervision,
            return_gt_zbuf=True)

        # Format input as batches to feed depth model
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

        close_fov_proxy_mask = torch.zeros(params.n_proxy_points, device=device).bool()

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
        
        navi_path = "/home/sli/MACARONS-main/data/rl_video_ash"
        

    # print("Trajectory computed in", time.time() - t0, "seconds.")
    # blender_X, blender_look = create_blender_curves(params, camera.X_cam_history, camera.V_cam_history,
    #   mirrored_pose=False)
    t2 = time.time()
    print("shijiannnnnn", t2-t1)
    print("Coverage Evolution:", coverage_evolution)
    
    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, gt_scene, surface_scene, full_pc, full_pc_colors


def test_random_walk_planning(params_file,
            model_file,
            results_json_file,
            numGPU,
            test_scenes,
            test_resolution=0.05,
            use_perfect_depth_map=False,
            compute_collision=False,
            load_json=False,
            dataset_path=None):

    params_path = os.path.join(configs_dir, params_file)
    weights_path = os.path.join(weights_dir, model_file,)
    results_json_path = os.path.join(results_dir, results_json_file)

    params = load_params(params_path)
    params.test_scenes = test_scenes
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

    # Setup model and dataloader, only get the test dataset
    dataloader, macarons, memory = setup_macarons_test(params, weights_path, device) 

    # Result json
    if load_json:
        with open(results_json_path, "r") as read_content:
            results_to_save = json.load(read_content)
    else:
        results_to_save = {}
    
    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict['occupied_pose']]
    
        for i_th_scene in range(len(scene_names)):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_th_scene]
            obj_name = obj_names[i_th_scene]
            settings = all_settings[i_th_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_th_scene] # Candidate camera poses
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

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])
            
            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
            trajectory_nb = memory.current_epoch % memory.n_trajectories
            training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
            training_poses_path = memory.get_poses_path(scene_memory_path)

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

                camera = setup_test_camera(params, mesh, start_cam_idx, settings, occupied_pose_data, gt_scene,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history, gt_scene, surface_scene, full_pc, full_pc_colors = compute_random_walk_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      device,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)
                

                # n_frames_to_recaputure = 400
                # for i in range(0, min(n_frames_to_recaputure, len(camera.X_cam_history))):
                #     X_cam = X_cam_history[i:i+1]
                #     V_cam = V_cam_history[i:i+1]

                #     fov_camera = camera.get_fov_camera_from_XV(X_cam, V_cam)
                #     camera.capture_image(mesh, fov_camera=fov_camera)

                plot_scene_and_tragectory_and_constructed_pt(scene_name=scene_name,params=params, gt_scene=gt_scene,
                                                            proxy_scene=proxy_scene, macarons=macarons,
                                                            surface_scene=surface_scene, camera=camera,
                                                            i_th_scene=i_th_scene, memory=memory,
                                                            device=device, results_dir=results_dir)
                
                
                results_to_save[scene_name][str(start_cam_idx_i)] = {}
                results_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                results_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                results_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                with open(results_json_path, 'w') as outfile:
                    json.dump(results_to_save, outfile)
                print("Saved data about test losses in", results_json_file)

                #-----------------------

    print("All trajectories computed.")

            








