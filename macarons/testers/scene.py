import os
import sys
import torch

import gc
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
from ..utility.long_term_utils import line_segment_intersects_point_cloud_region, check_camera_in_mesh
from navi.utility.utils import *
import json
import time
import ast

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")


def create_points_to_look_at(X_cam, V_cam, camera_size):
    rays = - get_cartesian_coords(r=torch.ones(len(V_cam), 1, device=V_cam.get_device()),
                                  elev=-1 * V_cam[:, 0].view(-1, 1),
                                  azim=180. + V_cam[:, 1].view(-1, 1),
                                  in_degrees=True)

    cam_pts = X_cam.view(-1, 3) + camera_size * rays
    cam_pts = cam_pts.view(-1, 3)

    return cam_pts


def convert_vector_to_blender(vec):
    new_vec = 0 + vec.cpu().numpy()
    alt_vec = np.copy(new_vec)
    new_vec[..., 1], new_vec[..., 2] = -alt_vec[..., 2], alt_vec[..., 1]
    return new_vec


def convert_blender_to_vector(vec):
    new_vec = np.array(vec)
    alt_vec = np.copy(new_vec)
    new_vec[..., 1], new_vec[..., 2] = alt_vec[..., 2], -alt_vec[..., 1]
    return new_vec


def create_blender_curves(params, X_cam_history, V_cam_history, cam_size=10, jump_poses=1, mirrored_pose=False):
    camera_X = convert_vector_to_blender(X_cam_history[params.n_interpolation_steps::jump_poses])
    camera_look = create_points_to_look_at(X_cam_history[params.n_interpolation_steps::jump_poses],
                                           V_cam_history[params.n_interpolation_steps::jump_poses],
                                           camera_size=cam_size * params.scene_scale_factor)
    camera_look = convert_vector_to_blender(camera_look)

    if mirrored_pose:
        camera_X[..., params.axis_to_mirror] = -1. * camera_X[..., params.axis_to_mirror]
        camera_look[..., params.axis_to_mirror] = -1. * camera_look[..., params.axis_to_mirror]

    camera_X = camera_X / params.scene_scale_factor
    camera_look = camera_look / params.scene_scale_factor

    return camera_X.tolist(), camera_look.tolist()


def setup_test(params, model_path, device):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path)
    print("\nThe following scenes will be used to test the model:")
    # for batch, elem in enumerate(test_dataloader):
    #     print(elem['scene_name'][0])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)

    trained_weights = torch.load(model_path, map_location=device)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True)  # todo: replace by params.ddp
    depth_losses = np.array(trained_weights["depth_losses"])
    depth_losses_per_epoch = (depth_losses[::2] + depth_losses[1::2]) / 2
    # depth_losses_per_epoch = depth_losses
    print("\nModel name:", model_path)
    print("\nThe model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")
    print("It has been trained for", trained_weights["epoch"], "epochs.")
    print("The loss was:", depth_losses_per_epoch[-1], depth_losses_per_epoch[-1] * 3 / 4)
    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Set loss functions
    pose_loss_fn = get_pose_loss_fn(params)
    regularity_loss_fn = get_regularity_loss_fn(params)
    ssim_loss_fn = None
    if params.training_mode == 'self_supervised':
        depth_loss_fn = get_reconstruction_loss_fn(params)
        ssim_loss_fn = get_ssim_loss_fn(params)
    else:
        raise NameError("Invalid training mode.")
    occ_loss_fn = get_occ_loss_fn(params)
    cov_loss_fn = get_cov_loss_fn(params)

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0)

    return test_dataloader, macarons, memory


def setup_test_scene(params,
                     mesh,
                     settings,
                     mirrored_scene,
                     device,
                     mirrored_axis=None,
                     surface_scene_feature_dim=1,
                     test_resolution=0.05,
                     covered_scene_feature_dim=1):
    """
    Setup the different scene objects used for prediction and performance evaluation.

    :param params:
    :param mesh:
    :param settings:
    :param device:
    :param is_master:
    :return:
    """

    # Initialize gt_scene: we use this scene to store gt surface points to evaluate the performance of the model.
    # This scene is not used for supervision during training, since the model is self-supervised from RGB data
    # captured in real-time.
    gt_scene = Scene(x_min=settings.scene.x_min-0.2,
                     x_max=settings.scene.x_max+0.2,
                     grid_l=settings.scene.grid_l,
                     grid_w=settings.scene.grid_w,
                     grid_h=settings.scene.grid_h,
                     cell_capacity=params.surface_cell_capacity,
                     cell_resolution=test_resolution * params.scene_scale_factor,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    covered_scene = Scene(x_min=settings.scene.x_min-0.2,
                          x_max=settings.scene.x_max+0.2,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=test_resolution * params.scene_scale_factor,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=covered_scene_feature_dim,
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface, gt_color = get_scene_gt_surface(gt_scene=gt_scene,
                                                         verts=mesh.verts_list()[0],
                                                         faces=mesh.faces_list()[0],
                                                         n_surface_points=params.n_gt_surface_points,
                                                         return_colors=True,
                                                         mesh=mesh)
    gt_scene.fill_cells(gt_surface, features=gt_color)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth model from RGB images
    surface_scene = Scene(x_min=settings.scene.x_min-0.2,
                          x_max=settings.scene.x_max+0.2,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=None,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=surface_scene_feature_dim,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points
    # settings.scene.x_min[[0, -1]] -= 3
    # settings.scene.x_max[[0, -1]] += 3

    proxy_scene = Scene(x_min=settings.scene.x_min-0.2,
                        x_max=settings.scene.x_max+0.2,
                        grid_l=settings.scene.grid_l ,
                        grid_w=settings.scene.grid_w,
                        grid_h=settings.scene.grid_h ,
                        cell_capacity=params.proxy_cell_capacity,
                        cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points,
                        device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1,  # We use proxy points indices as features
                        mirrored_scene=mirrored_scene,
                        score_threshold=params.score_threshold,
                        mirrored_axis=mirrored_axis)
    proxy_scene.initialize_proxy_points()

    # covered_scene = None
    # proxy_scene = None

    return gt_scene, covered_scene, surface_scene, proxy_scene


def setup_test_camera(params,
                      mesh, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      gt_scene,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=500000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test

    # Move to a valid neighbor pose before starting training.
    # Thus, we will have a few images to start training the depth module
    
    # flag_initial = False
    
    # while True:
    #     start_cam_idx = random.choice(list(camera.pose_space.keys()))
    #     start_cam_idx = torch.tensor(ast.literal_eval(start_cam_idx)).to(device)
        
    #     neighbor_indices = camera.get_neighboring_poses(pose_idx=start_cam_idx)
    #     valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
        
    #     for neigh in valid_neighbors:
    #         if not line_segment_intersects_point_cloud_region(gt_scene, camera.pose_space[str(neigh.cpu().numpy().tolist())][:3], camera.pose_space[str(start_cam_idx.cpu().numpy().tolist())][:3], device):
    #             first_cam_idx = neigh
    #             flag_initial = True
    #             break
        
    #     if flag_initial:
    #         break
        
    neighbor_indices = camera.get_neighboring_poses(pose_idx=start_cam_idx)
    valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

    for neigh in valid_neighbors:
        if not line_segment_intersects_point_cloud_region(gt_scene, camera.pose_space[str(neigh.cpu().numpy().tolist())][:3], camera.pose_space[str(start_cam_idx.cpu().numpy().tolist())][:3], device):
            first_cam_idx = neigh
            break

    # first_cam_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]
    # first_cam_idx = torch.tensor([0, 4, 1, 2, 2], device='cuda:1')
    # print("debug111", valid_neighbors)
    # print("debug111", neighbor_indices.size())
    # print("debug222",first_cam_idx)
    # Select a random, valid camera pose as starting pose
    print(first_cam_idx)
    first_cam_idx[3] = 2
    camera.initialize_camera(start_cam_idx=first_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(start_cam_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    return camera

def setup_training_2d_camera(params,
                      mesh, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      gt_scene,
                      proxy_scene,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=500000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test

    # Fixed the Y axis
    fixed_axes = (start_cam_idx[1], start_cam_idx[3])
    camera.pose_space = {
        k: v for k, v in camera.pose_space.items()
        if ast.literal_eval(k)[1] == fixed_axes[0] and ast.literal_eval(k)[3] == fixed_axes[1]
    }
    first_cam_idx = None
    while True:
    
        start_cam_idx = camera.get_random_valid_pose(mesh=mesh, proxy_scene=proxy_scene)

        neighbor_indices = camera.get_neighboring_poses_2d(pose_idx=start_cam_idx)
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        for neigh in valid_neighbors:
            if not line_segment_intersects_point_cloud_region(gt_scene, camera.pose_space[str(neigh.cpu().numpy().tolist())][:3], camera.pose_space[str(start_cam_idx.cpu().numpy().tolist())][:3], device):
                first_cam_idx = neigh
                break
        
        if first_cam_idx is not None:
            break

    # first_cam_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))]
    # first_cam_idx = torch.tensor([0, 4, 1, 2, 2], device='cuda:1')
    # print("debug111", valid_neighbors)
    # print("debug111", neighbor_indices.size())
    # print("debug222",first_cam_idx)
    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=first_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(start_cam_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    return camera
    
    
def setup_test_2d_camera(params,
                      mesh, mesh_for_check, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      gt_scene,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=500000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test


    # Fixed the Y axis and fillter the camere pose_space
        # Fixed the elevation to 2
    fixed_elevation = {key: val for key, val in camera.pose_space.items() if eval(key)[-2] == 2}
    camera.pose_space = fixed_elevation

    # Select a random camera pose as starting pose inside the 3d mesh
    next_idx_flag = True
    while next_idx_flag:
        pose_outside_mesh = True
        while pose_outside_mesh:
            random_pose_key = np.random.choice(list(camera.pose_space.keys()))
            random_pose = camera.pose_space[random_pose_key]

            # random_pose_point = np.array([[random_pose[0].cpu().numpy(), random_pose[1].cpu().numpy(), random_pose[2].cpu().numpy()]])
            # ray_directions = np.array([[0, -1, 0]])
            # inter_locations, _, _ = mesh_for_check.ray.intersects_location(ray_origins=random_pose_point,
            #                                                             ray_directions=ray_directions)
            # if len(inter_locations) % 2 == 1:
            #     start_cam_idx = camera.get_idx_from_key(random_pose_key)
            #     pose_outside_mesh = False
            random_pose_position = random_pose[:3]
            if check_camera_in_mesh(mesh_for_check, random_pose_position):
                start_cam_idx = camera.get_idx_from_key(random_pose_key)
                pose_outside_mesh = False

        # Select a random, valid camera pose as starting pose
        # start_cam_idx = camera.get_random_valid_pose(mesh=mesh, proxy_scene=proxy_scene)
        camera.cam_idx = start_cam_idx

        # Move to a valid neighbor pose before starting training.
        # Thus, we will have a few images to start training the depth module
        neighbor_indices = camera.get_neighboring_poses_2d()
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)
        
        # next_idx = valid_neighbors[np.random.randint(low=0, high=len(valid_neighbors))] Antoine's code
        for neigh in valid_neighbors:
            if not line_segment_mesh_intersection(camera.pose_space[str(neigh.cpu().numpy().tolist())][:3], 
                                                camera.pose_space[str(start_cam_idx.cpu().numpy().tolist())][:3], 
                                                mesh_for_check):
                next_idx = neigh
                next_idx_flag = False
                break

    camera.initialize_camera(start_cam_idx=start_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(next_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1
    return camera


def setup_test_iclr_camera(params,
                      mesh, mesh_for_check, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      gt_scene,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=500000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test


    # Fixed the Y axis and fillter the camere pose_space
        # Fixed the elevation to 2
    fixed_elevation = {key: val for key, val in camera.pose_space.items() if eval(key)[-2] == 2}
    camera.pose_space = fixed_elevation

    # Move to a valid neighbor pose before starting training.
    # Thus, we will have a few images to start training the depth module
    neighbor_indices = camera.get_neighboring_poses_2d(pose_idx=start_cam_idx)

    for neigh in neighbor_indices:
        if not line_segment_mesh_intersection(camera.pose_space[str(neigh.cpu().numpy().tolist())][:3], camera.pose_space[str(start_cam_idx.cpu().numpy().tolist())][:3], mesh_for_check):
            first_cam_idx = neigh
            break
    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=first_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    # We capture images along the way
    interpolation_step = 1
    for i in range(camera.n_interpolation_steps):
        camera.update_camera(start_cam_idx, interpolation_step=interpolation_step)
        camera.capture_image(mesh)
        interpolation_step += 1

    return camera


def compute_trajectory(params, macarons,
                       camera,
                       gt_scene, surface_scene, proxy_scene, covered_scene,
                       mesh,
                       device,
                       test_resolution=0.05,
                       use_perfect_depth_map=False,
                       compute_collision=False):

    macarons.eval()
    curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)
    curriculum_n_cells = get_curriculum_sampling_cell_number(params)

    full_pc = torch.zeros(0, 3, device=device)

    coverage_evolution = []
    t0 = time.time()

    for pose_i in range(params.n_poses_in_trajectory + 1):
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)

        # ----------Predict visible surface points from RGB images------------------------------------------------------

        # Load input RGB image and camera pose
        all_images, all_zbuf, all_mask, all_R, all_T, all_zfar = load_images_for_depth_model(camera=camera,
                                                                                             n_frames=1,
                                                                                             n_alpha=params.n_alpha,
                                                                                             return_gt_zbuf=True)
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

        if pose_i >= params.n_poses_in_trajectory:
            break

        surface_distance = curriculum_distances[pose_i]

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
            part_pc = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         mask=(mask * error_mask)[i:i + 1],
                                                         fov_cameras=camera.get_fov_camera_from_RT(
                                                             R_cam=batch_dict['R'][i:i + 1],
                                                             T_cam=batch_dict['T'][i:i + 1]),
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Fill surface scene
            part_pc_features = torch.zeros(len(part_pc), 1, device=device)
            surface_scene.fill_cells(part_pc, features=part_pc_features)

            full_pc = torch.vstack((full_pc, part_pc))

        # ----------Update Proxy Points data with current FoV-----------------------------------------------------------

        # Get Proxy Points in current FoV
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
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
        surface_scene.set_all_features_to_value(value=1.)

        # ----------Predict Occupancy Probability Field-----------------------------------------------------------------

        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(params, macarons.scone,
                                                                                           camera,
                                                                                           surface_scene, proxy_scene,
                                                                                           device)

        # ----------Predict Coverage Gain of neighbor camera poses------------------------------------------------------

        # Compute valid neighbor poses
        neighbor_indices = camera.get_neighboring_poses()
        valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

        max_coverage_gain = -1.
        next_idx = valid_neighbors[0]

        # For each valid neighbor...
        for neighbor_i in range(len(valid_neighbors)):
            neighbor_idx = valid_neighbors[neighbor_i]
            neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)

            # We check, if needed, if camera collides
            drop_neighbor = False
            if compute_collision:
                drop_neighbor = proxy_scene.camera_collides(params, camera, X_neighbor)

            if not drop_neighbor:
                # ...We predict its coverage gain...
                with torch.no_grad():
                    _, _, visibility_gains, coverage_gain = predict_coverage_gain_for_single_camera(
                        params=params, macarons=macarons.scone,
                        proxy_scene=proxy_scene, surface_scene=surface_scene,
                        X_world=X_world, proxy_view_harmonics=view_harmonics, occ_probs=occ_probs,
                        camera=camera, X_cam_world=X_neighbor, fov_camera=fov_neighbor)

                # ...And save it with the neighbor index if the estimated coverage is better
                if coverage_gain.shape[0] > 0 and coverage_gain > max_coverage_gain:
                    max_coverage_gain = coverage_gain
                    next_idx = neighbor_idx

        X_cam_t = 0. + camera.X_cam
        V_cam_t = 0. + camera.V_cam
        fov_camera_t = camera.get_fov_camera_from_XV(X_cam=X_cam_t, V_cam=V_cam_t)

        # ==============================================================================================================
        # Move to the neighbor NBV and acquire signal
        # ==============================================================================================================

        # Now that we have estimated the NBV among neighbors, we move toward this new camera pose and save RGB images
        # along the way.

        # ----------Move to next camera pose----------------------------------------------------------------------------
        # We move to the next pose and capture RGB images.
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
            part_pc = camera.compute_partial_point_cloud(depth=depth[i:i + 1],
                                                         mask=(mask * error_mask)[i:i + 1].bool(),
                                                         fov_cameras=fov_frame,
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Surface points to fill surface scene
            all_part_pc.append(part_pc)

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
            close_fov_proxy_mask[fov_proxy_mask] = False + (sgn_dists.view(-1).abs() < surface_distance)

        # ----------Update Scenes to finalize supervision signal and prepare next iteration-----------------------------

        # 1. Surface scene
        # Fill surface scene
        # We give a visibility=1 to points that were visible in frame t, and 0 to others
        complete_part_pc = torch.vstack(all_part_pc)
        complete_part_pc_features = torch.zeros(len(complete_part_pc), 1, device=device)
        complete_part_pc_features[:len(all_part_pc[0])] = 1.
        surface_scene.fill_cells(complete_part_pc, features=complete_part_pc_features)

        full_pc = torch.vstack((full_pc, complete_part_pc))

        # Compute coverage gain for each new camera pose
        # We also include, at the beginning, the previous camera pose with a coverage gain equal to 0.
        supervision_coverage_gains = torch.zeros(params.n_interpolation_steps, 1, device=device)
        for i in range(depth.shape[0]):
            supervision_coverage_gains[i, 0] = surface_scene.camera_coverage_gain(all_part_pc[i],
                                                                                  surface_epsilon=None)

        # Update visibility history of surface points
        surface_scene.set_all_features_to_value(value=1.)

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

    print("Trajectory computed in", time.time() - t0, "seconds.")
    # blender_X, blender_look = create_blender_curves(params, camera.X_cam_history, camera.V_cam_history,
    #   mirrored_pose=False)
    print("Coverage Evolution:", coverage_evolution)

    return coverage_evolution, camera.X_cam_history, camera.V_cam_history


def run_test(params_name,
             model_name,
             results_json_name,
             numGPU,
             test_scenes,
             test_resolution=0.05,
             use_perfect_depth_map=False,
             compute_collision=False,
             load_json=False,
             dataset_path=None):

    params_path = os.path.join(configs_dir, params_name)
    weights_path = os.path.join(weights_dir, model_name)
    results_json_path = os.path.join(results_dir, results_json_name)

    params = load_params(params_path)
    params.test_scenes = test_scenes
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
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

    # Setup device
    device = setup_device(params, None)

    # Setup model and dataloader
    dataloader, macarons, memory = setup_test(params, weights_path, device)

    # Result json
    if load_json:
        with open(results_json_path, "r") as read_content:
            dict_to_save = json.load(read_content)
    else:
        dict_to_save = {}

    print("\nModel path:", model_name)
    print("\nScore threshold:", params.score_threshold)

    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict['occupied_pose']]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            dict_to_save[scene_name] = {}

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None

            # Load segmented mesh
            # mesh = load_scene(segmented_mesh_path, params.scene_scale_factor, device, mirror=mirrored_scene)
            # gt_scene, _, _, _ = setup_test_scene(params,
            #                                      mesh,
            #                                      settings,
            #                                      mirrored_scene,
            #                                      device)

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
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                camera = setup_test_camera(params, mesh, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history = compute_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      device,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)

                dict_to_save[scene_name][str(start_cam_idx_i)] = {}
                dict_to_save[scene_name][str(start_cam_idx_i)]["coverage"] = coverage_evolution
                dict_to_save[scene_name][str(start_cam_idx_i)]["X_cam_history"] = X_cam_history.cpu().numpy().tolist()
                dict_to_save[scene_name][str(start_cam_idx_i)]["V_cam_history"] = V_cam_history.cpu().numpy().tolist()

                with open(results_json_path, 'w') as outfile:
                    json.dump(dict_to_save, outfile)
                print("Saved data about test losses in", results_json_name)

    print("All trajectories computed.")
