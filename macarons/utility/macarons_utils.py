import os
import random
import time
import torch
import torchvision
import shutil

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.datasets import collate_batched_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader
)
from pytorch3d.renderer.lighting import AmbientLights
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.renderer import MeshRenderer, HardFlatShader

from torchvision.transforms.functional import adjust_contrast
import torch.multiprocessing as mp

from .utils import (
    Params,
    floor_divide,
    sample_points_on_mesh_surface,
    init_weights,
    load_ddp_state_dict,
    NoamOpt,
    WarmupConstantOpt
)
from .spherical_harmonics import get_spherical_harmonics, clear_spherical_harmonics_cache
from .CustomGeometry import *
from .CustomDataset import SceneDataset

from ..networks.ManyDepth import (
    FeatureExtractor,
    ExpansionLayer,
    DisparityLayer,
    CostVolumeBuilder,
    DepthDecoder,
    PoseDecoder,
    ManyDepth,
)
from ..networks.SconeOcc import SconeOcc
from ..networks.SconeVis import SconeVis, KLDivCE, L1_loss, Uncentered_L1_loss
from ..networks.Macarons import Macarons, MacaronsWrapper, MacaronsOptimizer, create_macarons_model, create_macarons_depth


from .scone_utils import (
    compute_view_state,
    move_view_state_to_view_space,
    compute_view_harmonics,
    get_all_harmonics_under_degree,
    normalize_points_in_prediction_box,
    sample_proxy_points,
)
from .depth_model_utils import (
    transpose_channels,
    apply_jitter_to_images,
    apply_symmetry_to_images, hflip,
    get_relative_pose_matrices,
    convert_matrix_to_pose,
    compute_depth_from_disparity,
    compute_disparity_from_depth,
    get_pose_loss_fn,
    get_regularity_loss_fn,
    get_ssim_loss_fn,
    regularity_tab,
    pad,
)
from .idr_torch import rank as idr_torch_rank
from .idr_torch import size as idr_torch_size
from .idr_torch import local_rank as idr_torch_local_rank
from .idr_torch import cpus_per_task as idr_torch_cpus_per_task
from .idr_torch import hostnames as idr_torch_hostnames
from .idr_torch import gpus_ids as idr_torch_gpus_ids

def check_pixel_values(full_pc_projection, cam_img_position, size=10):
    height, width = full_pc_projection.shape[2], full_pc_projection.shape[3]
    
    x, y = cam_img_position
    
    x_min = max(x - size, 0)
    x_max = min(x + size + 1, height)
    y_min = max(y - size, 0)
    y_max = min(y + size + 1, width)
    
    region = full_pc_projection[0, 0, x_min:x_max, y_min:y_max]
    
    contains_one = (region == 1).any().item()
    
    return contains_one

def get_subfolder_names(folder_path):
    """
    Return a list of names of all subdirectories within a given directory.

    :param folder_path: The path to the directory from which subdirectories are listed.
    :return: List of subdirectory names.
    """
    subfolders = []  # List to store subfolder names
    # Use os.listdir to list all entries in the directory
    for entry in os.listdir(folder_path):
        # Construct the full path of the entry
        full_path = os.path.join(folder_path, entry)
        # Check if the entry is a directory and not a file
        if os.path.isdir(full_path):
            subfolders.append(entry)  # Add the directory name to the list
    
    return subfolders

def line_segment_mesh_intersection(start_point, end_point, mesh):
    """
    Compute the intersection between a line segment and a mesh.
    start_point: (torch.Tensor) Shape (3,)
    end_point: (torch.Tensor) Shape (3,)
    mesh: trimesh object
    """
    if not isinstance(start_point, list):
        start_point = start_point.cpu().numpy()
        end_point = end_point.cpu().numpy()
    else:
        start_point = np.array(start_point)
        end_point = np.array(end_point)
    direction = end_point - start_point
    line_length = np.linalg.norm(direction)
    direction = direction / line_length

    ray_origins = np.array([start_point])
    ray_directions = np.array([direction])

    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                   ray_directions=ray_directions)
    if locations.size > 0:
        distances = np.linalg.norm(locations - start_point, axis=1)
        intersection_on_segment = any(distances < line_length)

        if intersection_on_segment:
            return True
        else:
            return False
    else:
        return False

# axis_to_mirror = [0, 1, 2]

# ======================================================================================================================
# Roi model related functions
# ======================================================================================================================
# def load_pretrained_roi(pretrained_model_path, device, learn_pose=False, ddp=False):
#     # ---Method 1
#     # model = torch.load(pretrained_model_path, map_location=device).to(device)

#     print("\nCreating model...")
#     roi = create_roi_model(learn_pose=learn_pose, device=device)

#     print("\nLoading pretrained weights...")
#     weights_dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/macarons")
#     macarons.load_state_dict(torch.load(os.path.join(weights_dir_path, pretrained_model_path),
#                                         map_location=device), ddp=ddp)

#     return roi




# ======================================================================================================================

def setup_device(params, ddp_rank=None):

    if params.ddp:
        print("Setup device", str(ddp_rank), "for DDP training...")
        os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(params.WORLD_SIZE)
        os.environ['RANK'] = str(ddp_rank)

        dist.init_process_group("nccl", rank=ddp_rank, world_size=params.WORLD_SIZE)

        device = torch.device("cuda:" + str(ddp_rank))
        torch.cuda.set_device(device)

        torch.cuda.empty_cache()
        print("Setup done!")

        if ddp_rank == 0:
            print(torch.cuda.memory_summary())

    elif params.jz:
        print("Setup device", str(idr_torch_rank), " for Jean Zay training...")
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                rank=idr_torch_rank,
                                world_size=idr_torch_size)

        torch.cuda.set_device(idr_torch_local_rank)
        device = torch.device("cuda")

        torch.cuda.empty_cache()
        print("Setup done!")

        if idr_torch_rank == 0:
            print(torch.cuda.memory_summary())

    else:
        # Set our device:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(params.numGPU))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        print(device)

        # Empty cache
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())

    return device


def load_params(json_name, flatten=True):
    return Params(json_name, flatten=flatten)


def reduce_tensor(tensor: torch.Tensor, world_size):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def cleanup():
    dist.destroy_process_group()


def get_dataloader(train_scenes, val_scenes, test_scenes,
                   batch_size,
                   ddp=False, jz=False,
                   world_size=None, ddp_rank=None,
                   data_path=None,
                   use_occupied_pose=False):
    # Database path
    if data_path is None:
        if jz:
            database_path = "../../datasets/scenes/rgb"
        else:
            database_path = "../../../../datasets/scenes/rgb"
    else:
        database_path = data_path
    print("database_path----", database_path)

    # Dataset
    train_dataset = SceneDataset(data_path=database_path, scene_names=train_scenes, use_occupied_pose=use_occupied_pose)
    val_dataset = SceneDataset(data_path=database_path, scene_names=val_scenes, use_occupied_pose=use_occupied_pose)
    test_dataset = SceneDataset(data_path=database_path, scene_names=test_scenes, use_occupied_pose=use_occupied_pose)
    if ddp or jz:
        if jz:
            rank = idr_torch_rank
        else:
            rank = ddp_rank
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=True,
                                           drop_last=True)
        valid_sampler = DistributedSampler(val_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=False,
                                           drop_last=True)
        test_sampler = DistributedSampler(test_dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=False,
                                          drop_last=True)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      drop_last=True,
                                      collate_fn=collate_batched_meshes,
                                      sampler=train_sampler)
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           drop_last=True,
                                           collate_fn=collate_batched_meshes,
                                           sampler=valid_sampler)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     drop_last=True,
                                     collate_fn=collate_batched_meshes,
                                     sampler=test_sampler)
    else:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_batched_meshes,
                                      shuffle=True)
        validation_dataloader = DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_batched_meshes,
                                           shuffle=False)
        # train_dataloader = 0
        # validation_dataloader = 0
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_batched_meshes,
                                     shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def get_optimizer_single(params, model, learning_rate=None, warmup=None,
                         warmup_rate=None):
    if warmup is None:
        warmup = params.warmup
    if learning_rate is None:
        learning_rate = params.learning_rate
    if warmup_rate is None:
        warmup_rate = 1 / (warmup * learning_rate ** 2)

    if params.noam_opt:
        optimizer = NoamOpt(warmup_rate, warmup,
                            torch.optim.Adam(model.parameters(),
                                             lr=0,
                                             betas=(0.9, 0.98),
                                             eps=1e-9  # 1e-9
                                             )
                            )
        opt_name = "Noam"

    else:
        optimizer = WarmupConstantOpt(learning_rate=learning_rate,
                                      warmup=warmup,
                                      optimizer=torch.optim.AdamW(model.parameters(),
                                                                  lr=0
                                                                  )
                                      )
        # optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        opt_name = "WarmupAdamW"

    return optimizer, opt_name


def get_optimizer(params, model, previous_optimizer=None, depth_only=False):

    depth_optimizer, opt_name = get_optimizer_single(params, model.depth,
                                                     learning_rate=params.depth_learning_rate,
                                                     warmup=params.depth_warmup)

    if depth_only and previous_optimizer is not None:
        previous_optimizer.depth_optimizer = depth_optimizer
        return previous_optimizer, opt_name

    else:
        scone_optimizer, opt_name = get_optimizer_single(params, model.scone,
                                                         learning_rate=params.scone_learning_rate,
                                                         warmup=params.scone_warmup)
        optimizer = MacaronsOptimizer(depth_optimizer=depth_optimizer, scone_optimizer=scone_optimizer)

        return optimizer, opt_name


def update_learning_rate(params, optimizer, learning_rate):

    if params.noam_opt:
        optimizer.model_size = 1 / (optimizer.warmup * learning_rate ** 2)
        if optimizer._step == 0:
            optimizer._rate = 0
        else:
            optimizer._rate = (optimizer.model_size ** (-0.5)
                               * min(optimizer._step ** (-0.5),
                                     optimizer._step * optimizer.warmup ** (-1.5)))

    else:
        optimizer.learning_rate = learning_rate
        if optimizer._step == 0:
            optimizer._rate = 0
        else:
            optimizer._rate = optimizer.learning_rate * min(1., optimizer._step / optimizer.warmup)


def update_macarons_learning_rate(params, macarons_optimizer, depth_learning_rate, scone_learning_rate):
    update_learning_rate(params, macarons_optimizer.depth, depth_learning_rate)
    update_learning_rate(params, macarons_optimizer.scone, scone_learning_rate)
    

def load_pretrained_macarons(pretrained_model_path, device, learn_pose=False, ddp=False):
    # ---Method 1
    # model = torch.load(pretrained_model_path, map_location=device).to(device)

    print("\nCreating macarons model...")
    macarons = create_macarons_model(learn_pose=learn_pose, device=device)

    print("\nLoading pretrained macarons weights...")
    # weights_dir_path = "/home/sli/MACARONS-main/weights/macarons"
    # print(weights_dir_path)
    weights_dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/macarons")
    macarons.load_state_dict(torch.load(os.path.join(weights_dir_path, pretrained_model_path),
                                        map_location=device), ddp=ddp)

    return macarons


def initialize_macarons(params, macarons, device,
                        torch_seed=None, initialize=True,
                        pretrained=False, ddp_rank=None, find_unused_parameters=False,
                        return_training_data=False, checkpoint_name=None, previous_optimizer=None, depth_only=False,
                        load_from_ddp_model=True):
    """
    Initializes Macarons model for training.
    Can be initialized from scratch, or from an already trained model to resume training.

    :param params: params file.
    :param macarons: (Macarons)
    :param device: (device)
    :param torch_seed: (int)
    :param initialize: (bool)
    :param pretrained: (bool)
    :param ddp_rank: (int)
    :param find_unused_parameters: (bool)
    :param return_training_data: (bool)
    :param checkpoint_name: (str)
    :param previous_optimizer:
    :param depth_only: (bool)
    :param load_from_ddp_model: (bool)
    :return:
    """
    model_name = params.macarons_model_name
    start_epoch = 0
    best_loss = 1000.

    # Weight initialization if needed
    if initialize:
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            print("Seed", torch_seed, "chosen.")

            if not pretrained:
                macarons.apply(init_weights)

    # Load previous training weights if needed
    if not initialize:
        if checkpoint_name is None:
            checkpoint_name = "unvalidated_" + model_name + ".pth"
        weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../weights/macarons")
        checkpoint = torch.load(os.path.join(weights_dir, checkpoint_name), map_location=device)
        start_epoch = checkpoint['epoch']
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        else:
            best_loss = checkpoint['loss']

        ddp_model = False
        if load_from_ddp_model is None:
            if (model_name[:2] == "jz") or (model_name[:3] == "ddp"):
                ddp_model = True
        else:
            ddp_model = load_from_ddp_model

        #if ddp_model:
        #    macarons = load_ddp_state_dict(macarons, checkpoint['model_state_dict'])
        #else:
        macarons.load_state_dict(checkpoint['model_state_dict'], ddp_model, depth_only=depth_only)

    # DDP wrapping if needed
    if params.ddp:
        macarons.depth = torch.nn.SyncBatchNorm.convert_sync_batchnorm(macarons.depth)
        macarons.depth = DDP(macarons.depth, device_ids=[ddp_rank], find_unused_parameters=find_unused_parameters)
        if not depth_only:
            macarons.scone = DDP(macarons.scone, device_ids=[ddp_rank], find_unused_parameters=find_unused_parameters)
    elif params.jz:
        macarons.depth = torch.nn.SyncBatchNorm.convert_sync_batchnorm(macarons.depth)
        macarons.depth = DDP(macarons.depth,  device_ids=[idr_torch_local_rank],
                             find_unused_parameters=find_unused_parameters)
        if not depth_only:
            macarons.scone = DDP(macarons.scone,  device_ids=[idr_torch_local_rank],
                                 find_unused_parameters=find_unused_parameters)

    # Creating optimizer
    optimizer, opt_name = get_optimizer(params, macarons,
                                        previous_optimizer=previous_optimizer, depth_only=depth_only)
    if not initialize:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], depth_only=depth_only)

    if return_training_data:
        training_data_dict = {}
        training_data_dict['loss'] = checkpoint['loss']
        training_data_dict['train_losses'] = checkpoint['train_losses']
        training_data_dict['depth_losses'] = checkpoint['depth_losses']
        training_data_dict['occ_losses'] = checkpoint['occ_losses']
        training_data_dict['cov_losses'] = checkpoint['cov_losses']
        training_data_dict['train_coverages'] = checkpoint['train_coverages']

        return macarons, optimizer, opt_name, start_epoch, best_loss, training_data_dict

    return macarons, optimizer, opt_name, start_epoch, best_loss


# ======================================================================================================================
# Functions for Scene Management
# ======================================================================================================================
def load_scene_with_texture(mesh_path, scene_scale_factor, device, mirror=False, mirrored_axis=None):
    """
    Loads a 3D scene as a mesh, scales the vertices, and adds a red texture.

    :param mesh_path: (str)
    :param scene_scale_factor: (float)
    :param device:
    :param mirror: (bool) If True, mirrors the coordinates on x-axis.
    :param mirrored_axis: (list) List of axis to mirror.
    :return: Meshes object with red texture.
    """
    # Load the mesh
    mesh = load_objs_as_meshes([mesh_path], device=device)

    # Scale the vertices
    mesh.verts_list()[0] *= scene_scale_factor

    # Mirror the vertices if specified
    if mirror:
        if mirrored_axis is None:
            raise ValueError("Please provide the list of mirrored axes.")
        else:
            for axis in mirrored_axis:
                mesh.verts_list()[0][..., axis] = -mesh.verts_list()[0][..., axis]

    # Create a red texture
    texture_image = torch.ones(1, 3, 256, 256) * torch.tensor([1.0, 0.0, 0.0])

    # Create TexturesVertex object
    textures = TexturesVertex(verts_features=texture_image)

    # Add the texture to the mesh
    mesh.textures = textures

    return mesh
def load_scene(mesh_path, scene_scale_factor, device, mirror=False, mirrored_axis=None):
    """
    Loads a 3D scene as a mesh and scales the vertices.

    :param mesh_path: (str)
    :param scene_scale_factor: (float)
    :param device:
    :param mirror: (bool) If True, mirrors the coordinates on x-axis.
    :return:
    """
    mesh = load_objs_as_meshes([mesh_path], device=device)
    mesh.verts_list()[0] *= scene_scale_factor
    if mirror:
        if mirrored_axis is None:
            raise NameError("Please provide the list of mirrored axis.")
        else:
            for axis in mirrored_axis:
                mesh.verts_list()[0][..., axis] = -mesh.verts_list()[0][..., axis]
    return mesh

def load_scene_with_texture(mesh_path, scene_scale_factor, device, mirror=False, mirrored_axis=None):
    """
    Loads a 3D scene as a mesh, scales the vertices, and adds a red texture.

    :param mesh_path: (str)
    :param scene_scale_factor: (float)
    :param device:
    :param mirror: (bool) If True, mirrors the coordinates on x-axis.
    :param mirrored_axis: (list) List of axis to mirror.
    :return: Meshes object with red texture.
    """
    # Load the mesh
    mesh = load_objs_as_meshes([mesh_path], device=device)  

    # Scale the vertices
    mesh.verts_list()[0] *= scene_scale_factor

    # Mirror the vertices if specified
    if mirror:
        if mirrored_axis is None:
            raise ValueError("Please provide the list of mirrored axes.")
        else:
            for axis in mirrored_axis:
                mesh.verts_list()[0][..., axis] = -mesh.verts_list()[0][..., axis]

    # Create a red texture
    num_verts = mesh.verts_packed().shape[0]
    texture_image = torch.ones((1, num_verts, 3), device=device) * torch.tensor([0.5, 0.5, 0.5], device=device)

    # Create TexturesVertex object
    textures = TexturesVertex(verts_features=texture_image)

    # Add the texture to the mesh
    mesh.textures = textures

    return mesh


def get_scene_gt_surface(gt_scene, verts, faces, n_surface_points, return_colors=False,
                         mesh=None):
    """
    Return a GT surface point cloud for a scene, computed from verts and faces.
    The point cloud is filtered to that every point is inside the bounding box of the scene.
    Points are uniformly sampled on the surface of the mesh.

    :param gt_scene: (Scene)
    :param verts: (Tensor) Has shape (n_verts, 3)
    :param faces: (Tensor) Has shape (n_faces, 3)
    :param n_surface_points: (int)
    :return: (Tensor) Has shape (n_surface_points, 3)
    """
    _, inside_mask = gt_scene.get_pts_in_bounding_box(verts, return_mask=True)
    inside_faces = faces[torch.gather(input=inside_mask.view(-1, 1).expand(-1, 3), dim=0, index=faces).prod(-1).bool()]

    texture_face_indices = None
    if return_colors:
        texture_face_indices = torch.arange(len(faces), device=verts.get_device())[
            torch.gather(input=inside_mask.view(-1, 1).expand(-1, 3), dim=0, index=faces).prod(-1).bool()]
        
    gt_surface = sample_points_on_mesh_surface(verts, inside_faces, n_surface_points,
                                               return_colors=return_colors, mesh=mesh,
                                               texture_face_indices=texture_face_indices)

    return gt_surface


def create_scene_like(scene):
    """
    Create an empty scene with the same parameters as the provided scene.
    :param scene: (Scene)
    :return: (Scene)
    """
    new_scene = Scene(x_min=scene.x_min,
                      x_max=scene.x_max,
                      grid_l=scene.grid_l,
                      grid_w=scene.grid_w,
                      grid_h=scene.grid_h,
                      cell_capacity=scene.cell_capacity,
                      cell_resolution=scene.cell_resolution,
                      n_proxy_points=scene.n_proxy_points,
                      device=scene.device,
                      view_state_n_elev=scene.view_state_n_elev,
                      view_state_n_azim=scene.view_state_n_azim,
                      feature_dim=scene.feature_dim,
                      mirrored_scene=False)
    new_scene.mirrored_scene = scene.mirrored_scene
    new_scene.mirrored_axis = scene.mirrored_axis
    return new_scene


def create_scene_from_parameters(scene_parameters, device):
    """
    Create a scene object from a parameter dictionary.
    Such dictionaries are used during memory replay, and stored as .json in the Memory.

    :param scene_parameters: (dict)
    :param device: (Device)
    :return: (Scene)
    """
    scene = Scene(x_min=scene_parameters['x_min'],
                  x_max=scene_parameters['x_max'],
                  grid_l=scene_parameters['grid_l'],
                  grid_w=scene_parameters['grid_w'],
                  grid_h=scene_parameters['grid_h'],
                  cell_capacity=scene_parameters['cell_capacity'],
                  cell_resolution=scene_parameters['cell_resolution'],
                  n_proxy_points=scene_parameters['n_proxy_points'],
                  device=device,
                  view_state_n_elev=scene_parameters['view_state_n_elev'],
                  view_state_n_azim=scene_parameters['view_state_n_azim'],
                  feature_dim=scene_parameters['feature_dim'],
                  mirrored_scene=False)
    scene.mirrored_scene = scene_parameters['mirrored_scene']
    scene.mirrored_axis = scene_parameters['mirrored_axis']
    return scene


def fill_surface_scene(surface_scene, full_pc,
                       random_sampling_max_size=200000,
                       min_n_points_per_cell_fill=3,  # Should be higher if progressive_fill==False (maybe 100?)
                       progressive_fill=True,
                       max_n_points_per_fill=1000,  # Only used when progressive_fill==True
                       return_surface_points=False,
                       full_pc_colors=None):
    """
    Fill the provided scene object with a surface point cloud.
    A progressive fill method can be used to increase the quality of the reconstructed surface.

    :param surface_scene: (Scene) The scene to be filled.
    :param full_pc: (Tensor) Surface point cloud tensor with shape (N, 3).
    :param random_sampling_max_size: (int) To fill the scene object, we select a shuffled subset of the point cloud.
        Use this parameter to set a maximal size to this subset.
    :param min_n_points_per_cell_fill: (int) Minimal number of points to fill a cell of a scene.
    :param progressive_fill: (bool) Set this parameter to True to use the progressive filling method.
    :param max_n_points_per_fill: (int) Maximal number of points used at each step of progressive filling.
        Unused if progressive_fill is False.
    :param return_surface_points: (bool) If True, the function returns the final surface point cloud.
    :return: None if return_surface_points is False.
        Otherwise, a Tensor with shape (M, 3) representing the final surface point cloud.
    """
    # Selecting a random, shuffled subset of points
    sample_idx = torch.randperm(len(full_pc))[:random_sampling_max_size]
    sample_pc = full_pc[sample_idx]

    if full_pc_colors is None:
        sample_pc_features = torch.ones(len(sample_pc), 1, device=sample_pc.get_device())
    else:
        sample_pc_features = full_pc_colors[sample_idx]

    # Empty surface scene
    surface_scene.empty_cells()

    # If progressive_fill is False, the entire pc are directly added to the scene
    if not progressive_fill:
        surface_scene.fill_cells(sample_pc,
                                 features=sample_pc_features,
                                 n_point_min=min_n_points_per_cell_fill)  # n_point_min=100

    # If progressive_fill is True, the scene is progressively filled with points
    else:
        n_fill_to_do = random_sampling_max_size // max_n_points_per_fill
        if random_sampling_max_size % max_n_points_per_fill > 0:
            n_fill_to_do += 1

        for q_fill in range(n_fill_to_do):
            i_fill_inf = q_fill * max_n_points_per_fill
            i_fill_sup = i_fill_inf + max_n_points_per_fill
            if q_fill == random_sampling_max_size // max_n_points_per_fill:
                i_fill_sup = -1

            surface_scene.fill_cells(sample_pc[i_fill_inf:i_fill_sup],
                                     features=sample_pc_features[i_fill_inf:i_fill_sup],
                                     n_point_min=min_n_points_per_cell_fill)

    # surface_scene.set_all_features_to_value(value=1.)

    if return_surface_points:
        final_scene_pc = surface_scene.return_entire_pt_cloud(return_features=False)
        return final_scene_pc


def save_surface_scene_in_memory(surface_dir_path, surface_scene, surface_file_name='surface.pt'):
    """
    Save a surface scene to the provided path.

    :param surface_dir_path: (string) Path to the directory in which the scene will be saved.
    :param surface_scene: (Scene) Scene to be saved.
    :param surface_file_name: (string) Name of the file to be saved. Default name is 'surface.pt'.
    :return: None
    """
    dict_to_save = {}
    dict_to_save['surface_points'] = surface_scene.return_entire_pt_cloud(return_features=False)

    dict_to_save['scene_parameters'] = {}
    dict_to_save['scene_parameters']['x_min'] = surface_scene.x_min
    dict_to_save['scene_parameters']['x_max'] = surface_scene.x_max
    dict_to_save['scene_parameters']['grid_l'] = surface_scene.grid_l
    dict_to_save['scene_parameters']['grid_w'] = surface_scene.grid_w
    dict_to_save['scene_parameters']['grid_h'] = surface_scene.grid_h
    dict_to_save['scene_parameters']['cell_capacity'] = surface_scene.cell_capacity
    dict_to_save['scene_parameters']['cell_resolution'] = surface_scene.cell_resolution
    dict_to_save['scene_parameters']['n_proxy_points'] = surface_scene.n_proxy_points
    # dict_to_save['scene_parameters']['device']=surface_scene.device
    dict_to_save['scene_parameters']['view_state_n_elev'] = surface_scene.view_state_n_elev
    dict_to_save['scene_parameters']['view_state_n_azim'] = surface_scene.view_state_n_azim
    dict_to_save['scene_parameters']['feature_dim'] = surface_scene.feature_dim
    dict_to_save['scene_parameters']['mirrored_scene'] = surface_scene.mirrored_scene
    dict_to_save['scene_parameters']['mirrored_axis'] = surface_scene.mirrored_axis

    path_to_save = os.path.join(surface_dir_path, surface_file_name)
    torch.save(dict_to_save, path_to_save)


def save_occupancy_field_in_memory(occupancy_dir_path, proxy_scene, occupancy_file_name='occupancy_field.pt'):
    """
    Save a proxy scene to the provided path. Proxy points and their pseudo-GT occupancy values will be saved.
    On the contrary, out-of-field values and view state vectors will not be saved.

    :param occupancy_dir_path: (string) Path to the directory in which the scene will be saved.
    :param proxy_scene: (Scene) Scene to be saved.
    :param occupancy_file_name: (string) Name of the file to be saved. Default name is 'occupancy_field.pt'.
    :return: None
    """
    dict_to_save = {}
    dict_to_save['proxy_points'] = proxy_scene.proxy_points
    dict_to_save['proxy_probas'] = ((proxy_scene.proxy_supervision_occ > 0.) * (proxy_scene.out_of_field < 1.)).float()
    dict_to_save['proxy_n_inside_fov'] = proxy_scene.proxy_n_inside_fov
    dict_to_save['proxy_n_behind_depth'] = proxy_scene.proxy_n_behind_depth
    # todo: What about out of field?

    dict_to_save['scene_parameters'] = {}
    dict_to_save['scene_parameters']['x_min'] = proxy_scene.x_min
    dict_to_save['scene_parameters']['x_max'] = proxy_scene.x_max
    dict_to_save['scene_parameters']['grid_l'] = proxy_scene.grid_l
    dict_to_save['scene_parameters']['grid_w'] = proxy_scene.grid_w
    dict_to_save['scene_parameters']['grid_h'] = proxy_scene.grid_h
    dict_to_save['scene_parameters']['cell_capacity'] = proxy_scene.cell_capacity
    dict_to_save['scene_parameters']['cell_resolution'] = proxy_scene.cell_resolution
    dict_to_save['scene_parameters']['n_proxy_points'] = proxy_scene.n_proxy_points
    # dict_to_save['scene_parameters']['device']=proxy_scene.device
    dict_to_save['scene_parameters']['view_state_n_elev'] = proxy_scene.view_state_n_elev
    dict_to_save['scene_parameters']['view_state_n_azim'] = proxy_scene.view_state_n_azim
    dict_to_save['scene_parameters']['feature_dim'] = proxy_scene.feature_dim
    dict_to_save['scene_parameters']['mirrored_scene'] = proxy_scene.mirrored_scene
    dict_to_save['scene_parameters']['mirrored_axis'] = proxy_scene.mirrored_axis

    path_to_save = os.path.join(occupancy_dir_path, occupancy_file_name)
    torch.save(dict_to_save, path_to_save)


def load_surface_scene_from_memory(surface_dir_path, device, surface_file_name='surface.pt'):
    """
    Load a surface scene from the provided path.

    :param surface_dir_path: (string) Path to the directory from which the scene will be loaded.
    :param device: (Device)
    :param surface_file_name: (string) Name of the file to be loaded. Default name is 'surface.pt'.
    :return:
    """
    path_to_load = os.path.join(surface_dir_path, surface_file_name)
    surface_dict = torch.load(path_to_load, map_location=device)

    surface_points = surface_dict['surface_points']
    surface_features = torch.ones(len(surface_points), 1, device=device)

    surface_scene = create_scene_from_parameters(surface_dict['scene_parameters'], device)
    surface_scene.fill_cells(surface_points, features=surface_features)

    return surface_scene


def load_occupancy_field_from_memory(occupancy_dir_path, device, occupancy_file_name='occupancy_field.pt'):
    """
    Load a proxy scene from the provided path. Proxy points and their pseudo-GT occupancy values will be loaded.
    On the contrary, out-of-field values and view state vectors will are not provided.

    :param occupancy_dir_path: (string) Path to the directory from which the scene will be loaded.
    :param device: (Device)
    :param occupancy_file_name: (string) Name of the file to be loaded. Default name is 'occupancy_field.pt'.
    :return:
    """
    path_to_load = os.path.join(occupancy_dir_path, occupancy_file_name)
    occ_field_dict = torch.load(path_to_load, map_location=device)

    proxy_scene = create_scene_from_parameters(occ_field_dict['scene_parameters'], device)
    proxy_scene.initialize_proxy_points()

    proxy_scene.proxy_points = occ_field_dict['proxy_points']
    proxy_scene.proxy_supervision_occ = occ_field_dict['proxy_probas']
    proxy_scene.proxy_n_inside_fov = occ_field_dict['proxy_n_inside_fov']
    proxy_scene.proxy_n_behind_depth = occ_field_dict['proxy_n_behind_depth']
    # We keep oof at 1 such that the scene can be directly use for training iteration after filling it with a partial pc
    # proxy_scene.out_of_field = # We leave at 1.

    return proxy_scene


# ======================================================================================================================
# Functions for Camera Management
# ======================================================================================================================

def get_rgb_renderer_no_texture(image_height, image_width, ambient_light_intensity, cameras, device,
                     max_faces_per_bin=100000):
    
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        max_faces_per_bin=max_faces_per_bin,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = AmbientLights(ambient_color=((ambient_light_intensity,
                                           ambient_light_intensity,
                                           ambient_light_intensity),),
                           device=device)

    # Create a HardPhong renderer by composing a rasterizer and a shader
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer

def get_rgb_renderer(image_height, image_width, ambient_light_intensity, cameras, device,
                     max_faces_per_bin=100000):
    
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        # max_faces_per_bin=500000,
        max_faces_per_bin=max_faces_per_bin,
        # bin_size=50,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = AmbientLights(ambient_color=((ambient_light_intensity,
                                           ambient_light_intensity,
                                           ambient_light_intensity),),
                           device=device)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


def get_camera_RT(X_cam, V_cam):
    """
    Returns R, T matrices for corresponding cameras 3D positions X_cam and 2D directions V_cam.

    :param X_cam: (Tensor) Positions of camera centers. Tensor with shape (n_cam, 3)
    :param V_cam: (Tensor) Direction of cameras, as elevations and azimuths in degrees. Tensor with shape (n_cam, 2).
    :return: (Tuple of Tensors) R, T matrices of cameras.
        R_cam has shape (n_cam, 3, 3) and T_cam has shape (n_cam, 3).
    """
    rays = - get_cartesian_coords(r=torch.ones(len(V_cam), 1, device=V_cam.get_device()),
                                  elev=-1 * V_cam[:, 0].view(-1, 1),
                                  azim=180. + V_cam[:, 1].view(-1, 1),
                                  in_degrees=True)
    R_cam, T_cam = look_at_view_transform(eye=X_cam, at=X_cam + rays)
    if V_cam.get_device() > -1:
        R_cam, T_cam = R_cam.to(V_cam.get_device()), T_cam.to(V_cam.get_device())

    return R_cam, T_cam


# ======================================================================================================================
# Functions for Depth Model management
# ======================================================================================================================

def load_images_for_depth_model(camera, n_frames, n_alpha, frame_nb=None, frames_dir_path=None,
                                return_gt_zbuf=False):
    """
    Loads n_alpha images corresponding to the frames with number frame_nb and its previous frames.
    :param camera:
    :param n_frames:
    :param n_alpha:
    :param frame_nb:
    :return:
    """
    if frame_nb is None:
        current_frame_nb = camera.n_frames_captured - 1
    else:
        current_frame_nb = frame_nb
    all_images = torch.zeros(0, camera.image_height, camera.image_width, 3, device=camera.device)
    all_mask = torch.zeros(0, camera.image_height, camera.image_width, 1, device=camera.device)
    all_R = torch.zeros(0, 3, 3, device=camera.device)
    all_T = torch.zeros(0, 3, device=camera.device)
    all_zfar = torch.Tensor([camera.zfar]).to(camera.device).expand(n_frames + n_alpha)
    if return_gt_zbuf:
        all_zbuf = torch.zeros(0, camera.image_height, camera.image_width, 1, device=camera.device)

    if frames_dir_path is None:
        frames_dir_path = camera.save_dir_path

    for i_frame in range(n_frames + n_alpha):
        frame_nb = current_frame_nb - (n_frames + n_alpha - 1) + i_frame
        frame_path = os.path.join(frames_dir_path, str(frame_nb) + '.pt')
        frame_dict = torch.load(frame_path, map_location=camera.device)

        all_images = torch.cat((all_images, frame_dict['rgb']), dim=0)
        all_mask = torch.cat((all_mask, frame_dict['mask']), dim=0)
        all_R = torch.cat((all_R, frame_dict['R']), dim=0)
        all_T = torch.cat((all_T, frame_dict['T']), dim=0)
        if return_gt_zbuf:
            all_zbuf = torch.cat((all_zbuf, frame_dict['zbuf']), dim=0)

    if return_gt_zbuf:
        return all_images, all_zbuf, all_mask.bool(), all_R, all_T, all_zfar
    else:
        return all_images, all_mask.bool(), all_R, all_T, all_zfar


def create_batch_for_depth_model(params, all_images, all_mask, all_R, all_T, all_zfar, mode, device,
                                 all_zbuf=None):
    """

    :param params:
    :param all_images:
    :param all_mask:
    :param all_R:
    :param all_T:
    :param all_zfar:
    :param mode: (str) Can be "supervision" or "inference"
    :param device: (device)
    :param all_zbuf: (Tensor)
    :return:
    """
    start_idx = params.n_alpha
    end_idx = all_images.shape[0]

    if mode == 'supervision':
        n_alpha_to_gather = params.n_alpha_for_supervision
        if params.use_future_frame_for_supervision:
            end_idx -= 1
    else:
        n_alpha_to_gather = params.n_alpha

    alphas = params.alphas

    batch_images = all_images[start_idx:end_idx]
    batch_mask = all_mask[start_idx:end_idx]
    batch_R = all_R[start_idx:end_idx]
    batch_T = all_T[start_idx:end_idx]
    batch_zfar = all_zfar[start_idx:end_idx]
    if all_zbuf is not None:
        batch_zbuf = all_zbuf[start_idx:end_idx]

    batch_size = batch_images.shape[0]
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]

    alpha_images = torch.zeros(batch_size, 0, image_height, image_width, 3, device=device)
    alpha_mask = torch.zeros(batch_size, 0, image_height, image_width, 1, device=device)
    alpha_R = torch.zeros(batch_size, 0, 3, 3, device=device)
    alpha_T = torch.zeros(batch_size, 0, 3, device=device)
    alpha_zfar = batch_zfar.view(batch_size, 1).expand(-1, n_alpha_to_gather).contiguous()
    if all_zbuf is not None:
        alpha_zbuf = torch.zeros(batch_size, 0, image_height, image_width, 1, device=device)

    for i in range(n_alpha_to_gather):
        alpha = alphas[i]
        alpha_i_images = all_images[start_idx + alpha:end_idx + alpha].view(batch_size, 1, image_height, image_width, 3)
        alpha_i_mask = all_mask[start_idx + alpha:end_idx + alpha].view(batch_size, 1, image_height, image_width, 1)
        alpha_i_R = all_R[start_idx + alpha:end_idx + alpha].view(batch_size, 1, 3, 3)
        alpha_i_T = all_T[start_idx + alpha:end_idx + alpha].view(batch_size, 1, 3)

        alpha_images = torch.cat((alpha_images, alpha_i_images), dim=1)
        alpha_mask = torch.cat((alpha_mask, alpha_i_mask), dim=1)
        alpha_R = torch.cat((alpha_R, alpha_i_R), dim=1)
        alpha_T = torch.cat((alpha_T, alpha_i_T), dim=1)

        if all_zbuf is not None:
            alpha_i_zbuf = all_zbuf[start_idx + alpha:end_idx + alpha].view(batch_size, 1, image_height, image_width, 1)
            alpha_zbuf = torch.cat((alpha_zbuf, alpha_i_zbuf), dim=1)

    batch_dict = {'images': batch_images,
                  'mask': batch_mask.bool(),
                  'R': batch_R,
                  'T': batch_T,
                  'zfar': batch_zfar}

    alpha_dict = {'images': alpha_images,
                  'mask': alpha_mask.bool(),
                  'R': alpha_R,
                  'T': alpha_T,
                  'zfar': alpha_zfar}

    if all_zbuf is not None:
        batch_dict['zbuf'] = batch_zbuf
        alpha_dict['zbuf'] = alpha_zbuf

    return batch_dict, alpha_dict


def apply_depth_model(params, macarons, batch_dict, alpha_dict, device,
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
    else:
        pose, disp1, disp2, disp3, disp4 = macarons(mode='depth',
                                                    x=x, x_alpha=x_alpha[:, :params.n_alpha],
                                                    R=R, T=T, zfar=zfar,
                                                    device=device, gt_pose=gt_pose[:, :params.n_alpha])

        depth1 = compute_depth_from_disparity(params, disp1)
        depth2 = compute_depth_from_disparity(params, disp2)
        depth3 = compute_depth_from_disparity(params, disp3)
        depth4 = compute_depth_from_disparity(params, disp4)

        # Interpolating data at different scales
        depth2 = torch.nn.functional.interpolate(input=depth2, size=depth1.shape[-2:], mode='nearest')
        depth3 = torch.nn.functional.interpolate(input=depth3, size=depth1.shape[-2:], mode='nearest')
        depth4 = torch.nn.functional.interpolate(input=depth4, size=depth1.shape[-2:], mode='nearest')

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

    loss = 0.
    if compute_loss:
        if (depth_loss_fn is None) or (regularity_loss_fn is None) or (ssim_loss_fn is None) or (pose_loss_fn) is None:
            raise NameError("Please provide loss_fn arguments if you set compute_loss to True.")

        # Then we compute three losses: L2 pose loss, L1 loss on depth, regularity loss on disparity (maybe SSIM on depth?)
        disparity1 = compute_disparity_from_depth(params, depth1)
        disparity2 = compute_disparity_from_depth(params, depth2)
        disparity3 = compute_disparity_from_depth(params, depth3)
        disparity4 = compute_disparity_from_depth(params, depth4)

        pose_loss = pose_loss_fn(pose.contiguous(), gt_pose[:, :params.n_alpha].contiguous(),
                                 pose_factor=params.pose_factor, rotation_mode=params.rotation_mode)

        if params.regularity_loss and params.regularity_factor > 0.:
            mean_disp1 = disparity1.mean(2, True).mean(3, True)
            norm_disp1 = disparity1 / (mean_disp1 + 1e-7)
            norm_disp1[~mask1] *= 0.

            mean_disp2 = disparity2.mean(2, True).mean(3, True)
            norm_disp2 = disparity2 / (mean_disp2 + 1e-7)
            norm_disp2[~mask2] *= 0.

            mean_disp3 = disparity3.mean(2, True).mean(3, True)
            norm_disp3 = disparity3 / (mean_disp3 + 1e-7)
            norm_disp3[~mask3] *= 0.

            mean_disp4 = disparity4.mean(2, True).mean(3, True)
            norm_disp4 = disparity4 / (mean_disp4 + 1e-7)
            norm_disp4[~mask4] *= 0.

            regularity_loss = regularity_loss_fn(disp=norm_disp1, img=transposed_images) + \
                              regularity_loss_fn(disp=norm_disp2, img=transposed_images) * 1./2. + \
                              regularity_loss_fn(disp=norm_disp3, img=transposed_images) * 1./4. + \
                              regularity_loss_fn(disp=norm_disp4, img=transposed_images) * 1./8.

            regularity_loss = params.regularity_factor * regularity_loss

        else:
            regularity_loss = 0.


        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, zfar=zfar)
        alpha_cameras = FoVPerspectiveCameras(device=device,
                                              R=alpha_R.view(-1, 3, 3),
                                              T=alpha_T.view(-1, 3),
                                              zfar=params.zfar)
        depth_loss = depth_loss_fn(params=params,
                                   input_images=images, input_alpha_images=alpha_images, mask=mask,
                                   cameras=cameras, alpha_cameras=alpha_cameras,
                                   predicted_depth=transpose_channels(depth1, channel_is_at_the_end=False),
                                   macarons=macarons, ssim_loss_fn=ssim_loss_fn,
                                   channel_is_at_the_end=True, padding_mode=params.padding_mode) + \
                     depth_loss_fn(params=params,
                                   input_images=images, input_alpha_images=alpha_images, mask=mask,
                                   cameras=cameras, alpha_cameras=alpha_cameras,
                                   predicted_depth=transpose_channels(depth2, channel_is_at_the_end=False),
                                   macarons=macarons, ssim_loss_fn=ssim_loss_fn,
                                   channel_is_at_the_end=True, padding_mode=params.padding_mode) + \
                     depth_loss_fn(params=params,
                                   input_images=images, input_alpha_images=alpha_images, mask=mask,
                                   cameras=cameras, alpha_cameras=alpha_cameras,
                                   predicted_depth=transpose_channels(depth3, channel_is_at_the_end=False),
                                   macarons=macarons, ssim_loss_fn=ssim_loss_fn,
                                   channel_is_at_the_end=True, padding_mode=params.padding_mode) + \
                     depth_loss_fn(params=params,
                                   input_images=images, input_alpha_images=alpha_images, mask=mask,
                                   cameras=cameras, alpha_cameras=alpha_cameras,
                                   predicted_depth=transpose_channels(depth4, channel_is_at_the_end=False),
                                   macarons=macarons, ssim_loss_fn=ssim_loss_fn,
                                   channel_is_at_the_end=True, padding_mode=params.padding_mode)

        loss = pose_loss + depth_loss + regularity_loss
        loss = loss / 4.

    depth = depth1.detach()
    mask = mask1.detach()
    if symmetry_applied:
        depth = hflip(depth)
        mask = hflip(mask)
        error_mask = hflip(error_mask)

    if compute_loss:
        return loss, \
               transpose_channels(depth, channel_is_at_the_end=False), \
               transpose_channels(mask, channel_is_at_the_end=False).bool(), \
               transpose_channels(error_mask, channel_is_at_the_end=False).bool(), \
               pose.detach(), gt_pose.detach()

    else:
        return transpose_channels(depth, channel_is_at_the_end=False), \
               transpose_channels(mask, channel_is_at_the_end=False).bool(), \
               transpose_channels(error_mask, channel_is_at_the_end=False).bool(), \
               pose.detach(), gt_pose.detach()


def get_reconstruction_loss_fn(params):
    """

    :param params:
    :return:
    """
    def reconstruction_loss_fn(params,
                               input_images, input_alpha_images, mask,
                               cameras, alpha_cameras,
                               predicted_depth,
                               macarons,
                               ssim_loss_fn,
                               channel_is_at_the_end=True,
                               padding_mode='border'):
        """

        :param params:
        :param input_images:
        :param input_alpha_images:
        :param cameras:
        :param alpha_cameras:
        :param predicted_depth:
        :param depth_model:
        :param ssim_loss_fn:
        :param channel_is_at_the_end:
        :param padding_mode: (str) Padding mode for grid sampling during warping.
        Can be 'zeros', 'border' or 'reflection'.
        :return:
        """
        batch_size = input_images.shape[0]
        n_alpha = input_alpha_images.shape[1]
        if channel_is_at_the_end:
            height, width, n_channels = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            images = input_images
            alpha_images = input_alpha_images
        else:
            n_channels, height, width = input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            images = transpose_channels(input_images, channel_is_at_the_end=False)
            alpha_images = transpose_channels(input_alpha_images, channel_is_at_the_end=False)

        # Warping operation
        if params.jz or params.ddp:
            cost_volume_builder = macarons.module.depth.depth_decoder.cost_volume_builder
        else:
            cost_volume_builder = macarons.depth.depth_decoder.cost_volume_builder

        if params.use_depth_mask:
            warp_depth = 0. + predicted_depth
            warp_depth[~mask] = params.zfar
        else:
            warp_depth = predicted_depth

        world_points = cost_volume_builder.reproject_depth_map(warp_depth, cameras)
        world_points = world_points.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)

        warped_images = cost_volume_builder.warp(
            target_world_points=world_points.contiguous().view(-1, height, width, n_channels),
            source_features=alpha_images.view(-1, height, width, n_channels),
            source_cameras=alpha_cameras,
            features_channel_is_at_the_end=True,
            mode='bilinear',
            resize_target_to_fit_source=False,
            padding_mode=padding_mode).view(batch_size, n_alpha,
                                            height, width, n_channels)

        expanded_images = images.view(batch_size, 1, height, width, n_channels).expand(-1, n_alpha, -1, -1, -1)

        # Computing L1-loss
        l1_loss = torch.abs(expanded_images - warped_images).mean(-1, keepdim=True)

        # Computing, if needed, SSIM loss:
        if params.ssim_factor > 0:
            ssim_loss = ssim_loss_fn(
                transpose_channels(expanded_images.contiguous().view(-1, height, width, 3), channel_is_at_the_end=True),
                transpose_channels(warped_images.view(-1, height, width, 3), channel_is_at_the_end=True))

            ssim_loss = transpose_channels(ssim_loss, channel_is_at_the_end=False
                                           ).view(batch_size, n_alpha, height, width, n_channels
                                                  ).mean(-1, keepdim=True)

            loss = params.ssim_factor * ssim_loss + (1-params.ssim_factor) * l1_loss
        else:
            loss = l1_loss

        # Taking min over warped frames:
        loss = torch.min(loss, dim=1, keepdim=False)[0]

        # If using mask, we average only on masked pixels:
        if params.use_depth_mask:
            mask_factor = mask.sum(1, keepdim=True).sum(2, keepdim=True).expand(-1, height, width, -1) + 1e-7
            loss = torch.sum(loss * mask / mask_factor)
        else:
            loss = torch.mean(loss)

        return loss

    return reconstruction_loss_fn


# ======================================================================================================================
# Functions for Occupancy Model management
# ======================================================================================================================

def compute_occupancy_probability(macarons, pc, X, view_harmonics, mask=None,
                                  max_points_per_pass=20000):
    """

    :param macarons: (Macarons) MACARONS model.
    :param pc: (Tensor) Input point cloud tensor with shape (n_clouds, seq_len, pts_dim)
    :param X: (Tensor) Input query points tensor with shape (n_clouds, n_sample, x_dim)
    :param view_harmonics: (Tensor) View state harmonic features. Tensor with shape (n_clouds, seq_len, n_harmonics)
    :param max_points_per_pass: (int) Maximal number of points per forward pass.
    :return:
    """
    n_clouds, seq_len, pts_dim = pc.shape[0], pc.shape[1], pc.shape[2]
    n_sample, x_dim = X.shape[1], X.shape[2]
    n_harmonics = view_harmonics.shape[2]

    preds = torch.zeros(n_clouds, 0, 1).to(X.get_device())

    p = max_points_per_pass // n_clouds
    q = n_sample // p
    r = n_sample % p
    n_loop = q
    if r != 0:
        n_loop += 1

    for i in range(n_loop):
        low_idx = i * p
        up_idx = (i + 1) * p
        if i == q:
            up_idx = q * p + r
        preds_i = macarons(mode='occupancy',
                           partial_point_cloud=pc,
                           proxy_points=X[:, low_idx:up_idx],
                           view_harmonics=view_harmonics[:, low_idx:up_idx])
        preds_i = preds_i.view(n_clouds, up_idx - low_idx, -1)
        preds = torch.cat((preds, preds_i), dim=1)

    return preds


def compute_occupancy_probability_for_supervision(params, macarons, camera,
                                                  proxy_scene, proxy_mask,
                                                  surface_scene, n_cell_per_occ_forward_pass, device,
                                                  prediction_camera=None,
                                                  default_value=0.,
                                                  min_length=100):
    """
    Compute probability field of points near the surface, under a specified distance.
    This function is made for training supervision.

    :param params: (Params)
    :param macarons: (Macarons)
    :param camera: (Camera) Can be set to None if prediction_camera variable is not None.
    :param proxy_scene: (Scene)
    :param proxy_mask: (Tensor)
    :param surface_scene: (Scene)
    :param device:
    :param prediction_camera: (FoVPerspectiveCamera) If None, the current fov_camera from variable camera is used as
        the reference camera to compute the prediction.
    :param default_value: (float) Occupancy value given by default to the proxy points sampled for prediction, if no
        value has been predicted (which can occurs since the number of forward pass is set).
    :return:
    """

    # Sample indices (Random)
    if True:
        proxy_indices = proxy_scene.get_proxy_indices_from_mask(proxy_mask)
        proxy_indices = proxy_indices[torch.randperm(len(proxy_indices))[:params.n_proxy_point_for_occupancy_supervision]]

    # Sample indices (As many occupied points than empty points)
    # todo: Take as many occupied points than empty points
    if False:
        supervision_occupied_mask = proxy_scene.proxy_supervision_occ.view(-1) > 0
        occupied_indices = proxy_scene.get_proxy_indices_from_mask(proxy_mask * supervision_occupied_mask)
        empty_indices = proxy_scene.get_proxy_indices_from_mask(proxy_mask * (~supervision_occupied_mask))
        half_length = min(len(occupied_indices), len(empty_indices), params.n_proxy_point_for_occupancy_supervision // 2)
        if half_length > (min_length // 2):
            occupied_indices = occupied_indices[torch.randperm(len(occupied_indices))[:half_length]]
            empty_indices = empty_indices[torch.randperm(len(empty_indices))[:half_length]]
            proxy_indices = torch.cat((occupied_indices, empty_indices), dim=0)
        else:
            proxy_indices = proxy_scene.get_proxy_indices_from_mask(proxy_mask)
            proxy_indices = proxy_indices[torch.randperm(len(proxy_indices))[:params.n_proxy_point_for_occupancy_supervision]]

    # Get prediction mask from the sampled indices
    prediction_mask = proxy_scene.get_proxy_mask_from_indices(proxy_indices)
    proxy_points = proxy_scene.proxy_points[prediction_mask]

    # We sample proxy points
    # proxy_indices = torch.randperm(len(proxy_mask))[:params.n_proxy_point_for_occupancy_supervision]
    # proxy_points = proxy_scene.proxy_points[proxy_mask][proxy_indices]
    # occ_probs = torch.zeros(0, 1, device=device)

    # prediction_mask = torch.zeros_like(proxy_mask).bool()
    # prediction_mask[proxy_mask][proxy_indices] = True

    # Initialize their occupancy probability to default_value before computation
    # proxy_scene.proxy_proba[prediction_mask] = 0 # changed

    proxy_probas = torch.zeros_like(proxy_scene.proxy_proba)
    # proxy_probas = torch.zeros_like(proxy_scene.proxy_proba) + default_value


    # Get all cells containing such proxy points
    proxy_cells = proxy_scene.get_englobing_cells(proxy_points)

    # Get base harmonics values to allow spherical harmonics computation
    base_harmonics, h_polar, h_azim = get_all_harmonics_under_degree(params.harmonic_degree,
                                                                     params.view_state_n_elev,
                                                                     params.view_state_n_azim,
                                                                     device)

    # print("Occ: Len of proxy_cells:", len(proxy_cells))

    n_forward_pass = 0.
    proxy_cells_perm = torch.randperm(len(proxy_cells))

    for proxy_cell in proxy_cells[proxy_cells_perm]:
        if n_forward_pass >= n_cell_per_occ_forward_pass:
            break

        # Cell data
        cell_key = proxy_scene.get_key_from_idx(proxy_cell)
        cell_center = proxy_scene.cells[cell_key].center
        cell_diag = torch.linalg.norm(proxy_scene.cells[cell_key].x_max - proxy_scene.cells[cell_key].x_min)

        # Compute neighbor cells to proxy points...
        neighbor_cells = surface_scene.get_neighboring_cells(proxy_cell)
        # ...And get the associated partial surface point cloud
        cell_pc_world = surface_scene.get_pt_cloud_from_cells(neighbor_cells, return_features=False)

        # Compute the proxy points in the cell and the corresponding view state vectors
        _, cell_X_indices = proxy_scene.get_pt_cloud_from_cells(proxy_cell, return_features=True)
        cell_X_mask = proxy_scene.get_proxy_mask_from_indices(cell_X_indices)
        cell_X_mask = cell_X_mask * prediction_mask
        cell_X_world = proxy_scene.proxy_points[cell_X_mask]

        # If no prediction_camera is provided, predictions are made in current camera view space.
        # The camera space is centered on the cell and normalized in the neighborhood.
        if prediction_camera is None:
            if camera is None:
                raise NameError("Both camera and prediction_camera are equal to None.")
            prediction_camera = camera.fov_camera_0
        prediction_view_transform = prediction_camera.get_world_to_view_transform()
        prediction_box_center = prediction_view_transform.transform_points(cell_center.view(1, 3))
        prediction_box_diag = params.prediction_neighborhood_size * cell_diag

        # Predict Occupancy Probability
        if (cell_pc_world.shape[0] > 2*2*params.k_for_knn) and (len(cell_X_world) > 0):
            # try:
            # Move partial pc from world space to prediction view space, and normalize them in prediction box
            cell_pc = prediction_view_transform.transform_points(cell_pc_world)
            cell_pc = normalize_points_in_prediction_box(points=cell_pc,
                                                    prediction_box_center=prediction_box_center,
                                                    prediction_box_diag=prediction_box_diag).view(1, -1, 3)

            # Move proxy points from world space to prediction view space, and normalize them in prediction box
            cell_X = prediction_view_transform.transform_points(cell_X_world)
            cell_X = normalize_points_in_prediction_box(points=cell_X,
                                                        prediction_box_center=prediction_box_center,
                                                        prediction_box_diag=prediction_box_diag).view(1, -1, 3)

            # Compute view harmonics from view state vectors
            cell_view_states = move_view_state_to_view_space(proxy_scene.view_states[cell_X_mask].view(1,
                                                                               cell_X.shape[1],
                                                                               params.n_view_state_cameras),
                                                             prediction_camera,
                                                             n_elev=params.view_state_n_elev,
                                                             n_azim=params.view_state_n_azim)
            cell_view_harmonics = compute_view_harmonics(cell_view_states,
                                                         base_harmonics, h_polar, h_azim,
                                                         params.view_state_n_elev, params.view_state_n_azim)

            # Compute Occupancy Probability
            cell_occ_probs = compute_occupancy_probability(macarons=macarons,
                                                           pc=cell_pc, X=cell_X,
                                                           view_harmonics=cell_view_harmonics,
                                                           max_points_per_pass=20000).view(-1, 1)
            proxy_probas[cell_X_mask] += cell_occ_probs
            # proxy_probas[cell_X_mask] = 0. + cell_occ_probs  # CHANGED with the addition of default_value
            n_forward_pass += 1

    # If not enough forward passes have been made, we make dummy forward passes to let the ddp model train
    while n_forward_pass < n_cell_per_occ_forward_pass:
        dummy_X = proxy_scene.proxy_points[:params.k_for_knn + 1]
        dummy_pc = proxy_scene.proxy_points[:2 * 2 * params.k_for_knn + 1]
        dummy_harmonics = torch.zeros(len(dummy_X), params.n_harmonics, device=device)
        dummy_occ_probs = compute_occupancy_probability(macarons=macarons,
                                                        pc=dummy_pc.view(1, -1, 3), X=dummy_X.view(1, -1, 3),
                                                        view_harmonics=dummy_harmonics.view(1, -1, params.n_harmonics),
                                                        max_points_per_pass=20000).view(-1, 1) * 0.
        if n_forward_pass == 0:
            prediction_mask = torch.zeros(proxy_scene.n_proxy_points, device=device).bool()
            prediction_mask[:params.k_for_knn + 1] = True
            proxy_probas[prediction_mask] += 0. * dummy_occ_probs
        n_forward_pass += 1

    # print("N forward pass:", n_forward_pass)

    return prediction_mask, proxy_probas[prediction_mask]


def compute_scene_occupancy_probability_field(params, macarons, camera,
                                              surface_scene, proxy_scene, device,
                                              use_supervision_occ_mask=True,
                                              prediction_camera=None,
                                              use_supervision_occ_instead_of_predicted=False):
    """
    Compute occupancy probability field in the whole scene.
    This function is made for inference.

    :param params: (Params)
    :param macarons: (Macarons)
    :param camera: (Camera)
    :param surface_scene: (Scene)
    :param proxy_scene: (Scene)
    :param device:
    :return: X_world has shape (N, 3)
        view_harmonics has shape (N, n_view_harmonics)
        occ_probs has shape (N, 1)
    """
    # Initialize variables
    n_empty_cell = 0
    X_world = torch.zeros(0, 3, device=device)
    view_harmonics = torch.zeros(0, params.n_harmonics, device=device)
    # view_states = torch.zeros(0, params.view_state_n_elev * params.view_state_n_azim, device=device)
    occ_probs = torch.zeros(0, 1, device=device)

    # ----------Step 1--------------------------------------------------------------------------------------------------
    # Goal: We predict a probability for each proxy point that appeared in a camera field of view.

    # Get all proxy points that have been in the field of view of a camera, and are known not to be empty
    occ_mask = (proxy_scene.proxy_supervision_occ > 0.)[..., 0]
    all_fov_mask = (proxy_scene.out_of_field < 1.)[..., 0]
    if use_supervision_occ_mask:
        fovs_proxy_points = proxy_scene.proxy_points[occ_mask * all_fov_mask]
    else:
        fovs_proxy_points = proxy_scene.proxy_points[all_fov_mask]
    # Initialize their occupancy probability to zero before computation
    proxy_scene.proxy_proba[occ_mask * all_fov_mask] = 0.  # todo: Is 'occ_mask' necessary here?

    # Get all cells containing such proxy points
    proxy_cells = proxy_scene.get_englobing_cells(fovs_proxy_points)

    # Get base harmonics values to allow spherical harmonics computation
    base_harmonics, h_polar, h_azim = get_all_harmonics_under_degree(params.harmonic_degree,
                                                                     params.view_state_n_elev,
                                                                     params.view_state_n_azim,
                                                                     device)

    for proxy_cell in proxy_cells:
        # Cell data
        cell_key = proxy_scene.get_key_from_idx(proxy_cell)
        cell_center = proxy_scene.cells[cell_key].center
        cell_diag = torch.linalg.norm(proxy_scene.cells[cell_key].x_max - proxy_scene.cells[cell_key].x_min)

        # Compute neighbor cells to proxy points...
        neighbor_cells = surface_scene.get_neighboring_cells(proxy_cell)
        # ...And get the associated partial surface point cloud
        cell_pc_world = surface_scene.get_pt_cloud_from_cells(neighbor_cells, return_features=False)

        # Compute the proxy points in the cell and the corresponding view state vectors
        _, cell_X_indices = proxy_scene.get_pt_cloud_from_cells(proxy_cell, return_features=True)
        # Filter the proxy points using the pseudo-GT occupancy values
        if use_supervision_occ_mask:
            cell_X_mask = proxy_scene.get_proxy_mask_from_indices(cell_X_indices) * occ_mask
        else:
            cell_X_mask = proxy_scene.get_proxy_mask_from_indices(cell_X_indices)
        cell_X_world = proxy_scene.proxy_points[cell_X_mask]

        # Predictions are made in first camera view space, centered on the cell and normalized in the neighborhood
        if prediction_camera is None:
            if camera is None:
                raise NameError("Both camera and prediction_camera are equal to None.")
            prediction_camera = camera.fov_camera_0
        prediction_view_transform = prediction_camera.get_world_to_view_transform()
        prediction_box_center = prediction_view_transform.transform_points(cell_center.view(1, 3))
        prediction_box_diag = params.prediction_neighborhood_size * cell_diag

        # Predict Occupancy Probability
        if (cell_pc_world.shape[0] > 2*2*params.k_for_knn) and (len(cell_X_world) > 0):
            # try:
            # Move partial pc from world space to prediction view space, and normalize them in prediction box
            cell_pc = prediction_view_transform.transform_points(cell_pc_world)
            cell_pc = normalize_points_in_prediction_box(points=cell_pc,
                                                         prediction_box_center=prediction_box_center,
                                                         prediction_box_diag=prediction_box_diag).view(1, -1, 3)

            # Move proxy points from world space to prediction view space, and normalize them in prediction box
            cell_X = prediction_view_transform.transform_points(cell_X_world)
            cell_X = normalize_points_in_prediction_box(points=cell_X,
                                                        prediction_box_center=prediction_box_center,
                                                        prediction_box_diag=prediction_box_diag).view(1, -1, 3)

            # Compute view harmonics from view state vectors
            cell_view_states = move_view_state_to_view_space(proxy_scene.view_states[cell_X_mask.view(-1).bool()
                                                                                     ].view(1,
                                                                                            cell_X.shape[1],
                                                                                            params.n_view_state_cameras
                                                                                            ),
                                                             prediction_camera,
                                                             n_elev=params.view_state_n_elev,
                                                             n_azim=params.view_state_n_azim)
            cell_view_harmonics = compute_view_harmonics(cell_view_states,
                                                         base_harmonics, h_polar, h_azim,
                                                         params.view_state_n_elev, params.view_state_n_azim)

            # Compute Occupancy Probability
            if use_supervision_occ_instead_of_predicted:
                # cell_occ_probs = torch.ones(len(cell_X), 1, device=device)
                cell_occ_probs = proxy_scene.proxy_supervision_occ[cell_X_mask]
            else:
                cell_occ_probs = compute_occupancy_probability(macarons=macarons,
                                                               pc=cell_pc, X=cell_X,
                                                               view_harmonics=cell_view_harmonics,
                                                               max_points_per_pass=20000).view(-1, 1)  # todo: Increase max_points_per_pass?

            X_world = torch.vstack((X_world, cell_X_world.view(-1, 3)))
            view_harmonics = torch.vstack((view_harmonics, cell_view_harmonics.view(-1, params.n_harmonics)))
            occ_probs = torch.vstack((occ_probs, cell_occ_probs))

            # Update proba values in proxy scene
            proxy_scene.proxy_proba[cell_X_mask.view(-1).bool()] = cell_occ_probs

        else:
            n_empty_cell += 1

    # ----------Step 2--------------------------------------------------------------------------------------------------
    # Goal: We gather all points that stayed out of field, with a probability set to a default value (=0.5).
    oof_mask = (proxy_scene.out_of_field > 0.)[..., 0]

    oof_X_world = proxy_scene.proxy_points[oof_mask]
    oof_view_states = proxy_scene.view_states[oof_mask]
    # print("oof_view_state:", oof_view_states.shape)
    # oof_view_harmonics = compute_view_harmonics(oof_view_states.view(1,
    #                                                                  len(oof_X_world),
    #                                                                  params.n_view_state_cameras),
    #                                             base_harmonics, h_polar, h_azim,
    #                                             params.view_state_n_elev, params.view_state_n_azim)  # change this
    oof_view_harmonics = torch.zeros(1, len(oof_X_world), params.n_harmonics, device=device)
    # print("oof_view_harmonics:", oof_view_harmonics.shape)
    oof_occ_probs = proxy_scene.proxy_proba[oof_mask]

    X_world = torch.vstack((X_world, oof_X_world))
    view_harmonics = torch.vstack((view_harmonics, oof_view_harmonics.view(-1, params.n_harmonics)))
    occ_probs = torch.vstack((occ_probs, oof_occ_probs))

    return X_world, view_harmonics, occ_probs


def get_curriculum_sampling_distances(params, surface_scene, proxy_scene):
    min_surface_distance = 3 * proxy_scene.distance_between_proxy_points  # surface_scene.cell_resolution
    max_surface_distance = (2 * torch.linalg.norm(
        surface_scene.cells['[0, 0, 0]'].x_max - surface_scene.cells['[0, 0, 0]'].x_min)).item()

    x = np.arctan(10 * (np.linspace(0, 1, params.n_poses_in_trajectory) - 0.5))
    x -= np.min(x)
    x /= np.max(x)
    x = min_surface_distance + x * (max_surface_distance - min_surface_distance)

    return x


def get_curriculum_sampling_cell_number(params):
    min_cell_number = 5  # surface_scene.cell_resolution
    max_cell_number = 20

    n = min_cell_number + np.linspace(0, 1, params.n_poses_in_trajectory) * (max_cell_number - min_cell_number)
    n = np.floor(n)

    return n.astype(int)


def get_occ_loss_fn(params):
    if params.occ_loss_fn == "mse":
        occ_loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        return occ_loss_fn

    else:
        raise NameError("Invalid training loss function."
                        "Please choose a valid loss like 'mse'.")


# ======================================================================================================================
# Functions for Visibility Model management
# ======================================================================================================================

def mcts_predict_coverage_gain_for_single_camera(params, macarons,
                                            proxy_scene, surface_scene,
                                            X_world, proxy_view_harmonics, occ_probs,
                                            camera, X_cam_world, fov_camera,
                                            prediction_camera=None, device=None):
    """

    :param params:
    :param macarons:
    :param proxy_scene:
    :param X_world: (Tensor) Has shape (N, 3)
    :param proxy_view_harmonics: (Tensor) Has shape (N, n_harmonics)
    :param occ_probs: (Tensor) Has shape (N, 1)
    :param camera: (Camera) Can be set to None if prediction_camera variable is not None.
    :param X_cam_world:
    :param fov_camera:
    :param prediction_camera: (FoVPerspectiveCamera) If None, the current fov_camera from variable camera is used as
        the reference camera to compute the prediction.
    :return:
    """

    # We compute proxy points in camera fov
    fov_X_world, fov_X_mask = camera.get_points_in_fov(X_world, return_mask=True,
                                                       fov_camera=fov_camera,
                                                       fov_range=params.sensor_range)
    fov_view_harmonics = proxy_view_harmonics[fov_X_mask]
    fov_occ_probs = occ_probs[fov_X_mask]

    # We remove proxy points with very low occupancy probability
    occ_mask = fov_occ_probs[..., 0] > params.min_occ_for_proxy_points

    fov_X_world = fov_X_world[occ_mask]
    fov_view_harmonics = fov_view_harmonics[occ_mask]
    fov_occ_probs = fov_occ_probs[occ_mask]
    
    forward_pass_computed = False


    # print("Coverage debug marker 2:", fov_X_world.shape, camera.X_cam)

    if len(fov_X_world) > 0:
        # N_proxy_points_in_fov = len(fov_X_world)
        fov_proxy_volume = fov_occ_probs.sum()

        # We sample proxy points
        proxy_points, view_harmonics, sample_idx = sample_proxy_points(fov_X_world, fov_occ_probs, fov_view_harmonics,
                                                                       n_sample=params.seq_len,
                                                                       min_occ=params.min_occ_for_proxy_points,
                                                                       use_occ_to_sample=params.use_occ_to_sample_proxy_points,
                                                                       return_index=True)
        proxy_points_world = 0. + proxy_points
        # print("Coverage debug marker 3:", proxy_points.shape, camera.X_cam)

        # Initialize prediction box
        proxy_points_center = (proxy_points[..., :3].max(dim=0, keepdim=True)[0]
                               + proxy_points[..., :3].min(dim=0, keepdim=True)[0]).view(1, 3) / 2.

        if prediction_camera is None:
            if camera is None:
                raise NameError("Both camera and prediction_camera are equal to None.")
            prediction_camera = camera.fov_camera_0

        prediction_view_transform = prediction_camera.get_world_to_view_transform()
        prediction_box_center = prediction_view_transform.transform_points(proxy_points_center)
        prediction_box_diag = torch.linalg.norm(proxy_scene.x_max - proxy_scene.x_min).item()  # todo: Maybe change this. User should be able to choose the size?
        # print("Coverage debug marker 4:", prediction_box_diag, camera.X_cam)

        # Move proxy points from world space to prediction view space, and normalize them in prediction box
        proxy_points[..., :3] = prediction_view_transform.transform_points(proxy_points[..., :3])
        proxy_points[..., :3] = normalize_points_in_prediction_box(points=proxy_points[..., :3],
                                                                   prediction_box_center=prediction_box_center,
                                                                   prediction_box_diag=prediction_box_diag)
        proxy_points = torch.unsqueeze(proxy_points, dim=0)
        view_harmonics = torch.unsqueeze(view_harmonics, dim=0)
        # print("Coverage debug marker 5:", view_harmonics.shape, camera.X_cam)

        # Move camera coordinates from world space to prediction view space, and normalize them for prediction box
        X_cam = prediction_view_transform.transform_points(X_cam_world)
        X_cam = normalize_points_in_prediction_box(points=X_cam,
                                                   prediction_box_center=prediction_box_center,
                                                   prediction_box_diag=prediction_box_diag)
        X_cam = torch.unsqueeze(X_cam, dim=0)
        # print("Coverage debug marker 6:", X_cam.shape, camera.X_cam)

        # Predict visibility gain harmonics
        visibility_gain_harmonics = macarons(mode='visibility',
                                             proxy_points=proxy_points, view_harmonics=view_harmonics)
        forward_pass_computed = True
        # print("Coverage debug marker 7:", visibility_gain_harmonics.shape, camera.X_cam)

        # Use MC sampling indices to retrieve the sampled proxy points with duplicates
        proxy_points = torch.unsqueeze(proxy_points[0][sample_idx], dim=0)
        proxy_points_world = torch.unsqueeze(proxy_points_world[sample_idx], dim=0)
        visibility_gain_harmonics = torch.unsqueeze(visibility_gain_harmonics[0][sample_idx], dim=0)
        view_harmonics = torch.unsqueeze(view_harmonics[0][sample_idx], dim=0)
        # print("Coverage debug marker 8:", visibility_gain_harmonics.shape, camera.X_cam)

        visibility_gains = macarons.compute_visibility_gains(pts=proxy_points,
                                                                 harmonics=visibility_gain_harmonics,
                                                                 X_cam=X_cam)

        # distance_factor_th is 17,  Threshold used for penalizing the distance to the camera when predicting surface coverage gains.
        visibility_gains = visibility_gains * get_distance_factor_threshold(pts=proxy_points_world[..., :3].view(-1, 3),
                                                                                X_cam=X_cam_world,
                                                                                distance_th=params.distance_factor_th
                                                                                ).view(1, 1, -1)
        # Finally, compute coverage gain
        coverage_gain = torch.mean(visibility_gains, dim=-1) * fov_proxy_volume

    else:

        fov_proxy_volume = None
        visibility_gains = None
        proxy_points_world = None
        view_harmonics = None
        fov_X_world = None
        sample_idx = None
        fov_occ_probs = None
        coverage_gain = torch.zeros((1, 1)).to(device)
    
    # max_value, max_index = torch.max(sample_idx, dim=0)
    # print("debug111...", max_value.item())
    # print("debug111.", fov_X_mask.size())
    # print("debug", fov_occ_probs.size())
    # print("dedef---", fov_view_harmonics.size())

    return sample_idx, fov_occ_probs, fov_X_world, fov_X_mask, visibility_gains, coverage_gain.view(-1, 1)

def predict_coverage_gain_for_single_camera(params, macarons,
                                            proxy_scene, surface_scene,
                                            X_world, proxy_view_harmonics, occ_probs,
                                            camera, X_cam_world, fov_camera,
                                            prediction_camera=None, device=None):
    """

    :param params:
    :param macarons:
    :param proxy_scene:
    :param X_world: (Tensor) Has shape (N, 3)
    :param proxy_view_harmonics: (Tensor) Has shape (N, n_harmonics)
    :param occ_probs: (Tensor) Has shape (N, 1)
    :param camera: (Camera) Can be set to None if prediction_camera variable is not None.
    :param X_cam_world:
    :param fov_camera:
    :param prediction_camera: (FoVPerspectiveCamera) If None, the current fov_camera from variable camera is used as
        the reference camera to compute the prediction.
    :return:
    """
    # print("Coverage debug marker 1:", camera.X_cam)

    # We compute proxy points in camera fov
    fov_X_world, fov_X_mask = camera.get_points_in_fov(X_world, return_mask=True,
                                                       fov_camera=fov_camera,
                                                       fov_range=params.sensor_range)
    fov_view_harmonics = proxy_view_harmonics[fov_X_mask]
    fov_occ_probs = occ_probs[fov_X_mask]
    count = fov_X_mask.sum().item()

    # We remove proxy points with very low occupancy probability
    occ_mask = fov_occ_probs[..., 0] > params.min_occ_for_proxy_points

    fov_X_world = fov_X_world[occ_mask]
    fov_view_harmonics = fov_view_harmonics[occ_mask]
    fov_occ_probs = fov_occ_probs[occ_mask]

    forward_pass_computed = False

    # print("Coverage debug marker 2:", fov_X_world.shape, camera.X_cam)

    if len(fov_X_world) > 0:
        # N_proxy_points_in_fov = len(fov_X_world)
        fov_proxy_volume = fov_occ_probs.sum()

        # We sample proxy points
        proxy_points, view_harmonics, sample_idx = sample_proxy_points(fov_X_world, fov_occ_probs, fov_view_harmonics,
                                                                       n_sample=params.seq_len,
                                                                       min_occ=params.min_occ_for_proxy_points,
                                                                       use_occ_to_sample=params.use_occ_to_sample_proxy_points,
                                                                       return_index=True)
        proxy_points_world = 0. + proxy_points
        # print("Coverage debug marker 3:", proxy_points.shape, camera.X_cam)

        # Initialize prediction box
        proxy_points_center = (proxy_points[..., :3].max(dim=0, keepdim=True)[0]
                               + proxy_points[..., :3].min(dim=0, keepdim=True)[0]).view(1, 3) / 2.

        if prediction_camera is None:
            if camera is None:
                raise NameError("Both camera and prediction_camera are equal to None.")
            prediction_camera = camera.fov_camera_0

        prediction_view_transform = prediction_camera.get_world_to_view_transform()
        prediction_box_center = prediction_view_transform.transform_points(proxy_points_center)
        prediction_box_diag = torch.linalg.norm(proxy_scene.x_max - proxy_scene.x_min).item()  # todo: Maybe change this. User should be able to choose the size?
        # print("Coverage debug marker 4:", prediction_box_diag, camera.X_cam)

        # Move proxy points from world space to prediction view space, and normalize them in prediction box
        proxy_points[..., :3] = prediction_view_transform.transform_points(proxy_points[..., :3])
        proxy_points[..., :3] = normalize_points_in_prediction_box(points=proxy_points[..., :3],
                                                                   prediction_box_center=prediction_box_center,
                                                                   prediction_box_diag=prediction_box_diag)
        proxy_points = torch.unsqueeze(proxy_points, dim=0)
        view_harmonics = torch.unsqueeze(view_harmonics, dim=0)
        # print("Coverage debug marker 5:", view_harmonics.shape, camera.X_cam)

        # Move camera coordinates from world space to prediction view space, and normalize them for prediction box
        X_cam = prediction_view_transform.transform_points(X_cam_world)
        X_cam = normalize_points_in_prediction_box(points=X_cam,
                                                   prediction_box_center=prediction_box_center,
                                                   prediction_box_diag=prediction_box_diag)
        X_cam = torch.unsqueeze(X_cam, dim=0)
        # print("Coverage debug marker 6:", X_cam.shape, camera.X_cam)

        # Predict visibility gain harmonics
        visibility_gain_harmonics = macarons(mode='visibility',
                                             proxy_points=proxy_points, view_harmonics=view_harmonics)
        forward_pass_computed = True
        # print("Coverage debug marker 7:", visibility_gain_harmonics.shape, camera.X_cam)

        # Use MC sampling indices to retrieve the sampled proxy points with duplicates
        proxy_points = torch.unsqueeze(proxy_points[0][sample_idx], dim=0)
        proxy_points_world = torch.unsqueeze(proxy_points_world[sample_idx], dim=0)
        visibility_gain_harmonics = torch.unsqueeze(visibility_gain_harmonics[0][sample_idx], dim=0)
        view_harmonics = torch.unsqueeze(view_harmonics[0][sample_idx], dim=0)
        # print("Coverage debug marker 8:", visibility_gain_harmonics.shape, camera.X_cam)

        # Compute per-point visibility gains
        if params.jz or params.ddp:
            visibility_gains = macarons.module.compute_visibility_gains(pts=proxy_points,
                                                                        harmonics=visibility_gain_harmonics,
                                                                        X_cam=X_cam)
            # print("Coverage debug marker 9:", visibility_gains.shape, camera.X_cam)
        else:
            visibility_gains = macarons.compute_visibility_gains(pts=proxy_points,
                                                                    harmonics=visibility_gain_harmonics,
                                                                    X_cam=X_cam)

        # distance_factor_th is 17,  Threshold used for penalizing the distance to the camera when predicting surface coverage gains.
        if params.distance_factor_th is None:
            visibility_gains = visibility_gains * get_distance_factor(params, proxy_points_world[..., :3].view(-1, 3),
                                                                      X_cam_world, fov_camera,
                                                                      surface_scene.cell_resolution).view(1, 1, -1)
        elif params.distance_factor_th == 'smooth':
            visibility_gains = visibility_gains * get_distance_factor_smooth(params=params,
                                                                             pts=proxy_points_world[..., :3].view(-1, 3),
                                                                             X_cam=X_cam_world,
                                                                             fov_camera=fov_camera,
                                                                             cell_resolution=surface_scene.cell_resolution
                                                                             ).view(1, 1, -1)
        else:
            visibility_gains = visibility_gains * get_distance_factor_threshold(pts=proxy_points_world[..., :3].view(-1, 3),
                                                                                X_cam=X_cam_world,
                                                                                distance_th=params.distance_factor_th
                                                                                ).view(1, 1, -1)
        # Finally, compute coverage gain
        coverage_gain = torch.mean(visibility_gains, dim=-1) * fov_proxy_volume
        # print("coverage_gains=========")
        # print(coverage_gain[0].item())
        # print("Coverage debug marker 10:", coverage_gain, camera.X_cam)

    else:
        print("Cov: No forward pass computed. Creating dummy input.", camera.X_cam)
        device = camera.device
        # proxy_points_world = torch.zeros(1, 0, 3, device=device)
        # view_harmonics = torch.zeros(1, 0, params.n_harmonics, device=device)
        # visibility_gains = torch.zeros(1, 1, 0, device=device)
        # coverage_gain = torch.zeros(0, 1, device=device)

        dummy_proxy_points = torch.zeros(1, params.k_for_knn, 4, device=device)
        dummy_view_harmonics = torch.zeros(1, params.k_for_knn, params.n_harmonics, device=device)

        # Predict visibility gain harmonics
        dummy_visibility_gain_harmonics = macarons(mode='visibility',
                                                   proxy_points=dummy_proxy_points,
                                                   view_harmonics=dummy_view_harmonics)

        # Compute per-point visibility gains
        if params.jz or params.ddp:
            visibility_gains = macarons.module.compute_visibility_gains(pts=dummy_proxy_points,
                                                                        harmonics=dummy_visibility_gain_harmonics,
                                                                        X_cam=X_cam_world.view(1, -1, 3))
            # print("Coverage debug marker 9:", visibility_gains.shape, camera.X_cam)
        else:
            visibility_gains = macarons.compute_visibility_gains(pts=dummy_proxy_points,
                                                                 harmonics=dummy_visibility_gain_harmonics,
                                                                 X_cam=X_cam_world.view(1, -1, 3))
        # Finally, compute coverage gain
        coverage_gain = torch.mean(visibility_gains, dim=-1) * 0.
        proxy_points_world = dummy_proxy_points
        view_harmonics = dummy_view_harmonics

        fov_proxy_volume = None
        # visibility_gains = None
        # proxy_points_world = None
        # view_harmonics = None
        # coverage_gain = torch.zeros((1, 1)).to(device)

    return proxy_points_world, view_harmonics, fov_proxy_volume, visibility_gains, coverage_gain.view(-1, 1)


def get_distance_factor(params, pts, X_cam, fov_camera, cell_resolution):
    """

    :param params:
    :param pts: (Tensor) Has shape (n_points, 3)
    :param X_cam: (Tensor) Has shape (1, 3)
    :param fov_camera:
    :param cell_resolution: (float)
    :return:
    """
    n_pts = pts.shape[0]
    focal_length = 1. / torch.tan(np.pi / 180. * fov_camera.fov / 2.)
    pixel_size = 2. / min(params.image_height, params.image_width)
    epsilon = np.sqrt(np.pi) / 2. * cell_resolution
    distance_th = focal_length * epsilon / pixel_size

    dists = torch.linalg.norm(pts - X_cam.view(1, 3), dim=-1, keepdim=True)
    dist_mask = dists > distance_th

    res = torch.ones(n_pts, 1, device=pts.get_device())
    # res[dist_mask] = distance_th * distance_th / torch.pow(dists[dist_mask], exponent=2)

    res[dist_mask] = epsilon ** 2 * (focal_length / pixel_size / dists[dist_mask]) ** 2

    return res


def get_distance_factor_threshold(pts, X_cam, distance_th=17.):
    n_pts = pts.shape[0]
    dists = torch.linalg.norm(pts - X_cam.view(1, 3), dim=-1, keepdim=True)
    dist_mask = dists > distance_th

    res = torch.ones(n_pts, 1, device=pts.get_device())
    res[dist_mask] *= distance_th ** 2 / (dists[dist_mask]) ** 2

    return res


def get_distance_factor_smooth(params, pts, X_cam, fov_camera, cell_resolution):
    focal_length = 1. / torch.tan(np.pi / 180. * fov_camera.fov / 2.)
    pixel_size = 2. / min(params.image_height, params.image_width)
    epsilon = np.sqrt(np.pi) / 2. * cell_resolution
    distance_th = focal_length * epsilon / pixel_size

    dists = torch.linalg.norm(pts - X_cam.view(1, 3), dim=-1, keepdim=True)
    res = 1. / (1. + (dists / distance_th) ** 2)

    return res


def get_cov_loss_fn(params):
    if params.cov_loss_fn == "kl_divergence":
        cov_loss_fn = KLDivCE()

    elif params.cov_loss_fn == "l1":
        cov_loss_fn = L1_loss()

    elif params.cov_loss_fn == "uncentered_l1":
        cov_loss_fn = Uncentered_L1_loss()

    else:
        raise NameError("Invalid training loss function."
                        "Please choose a valid loss between 'kl_divergence', 'l1' or 'uncentered_l1.")

    return cov_loss_fn


# ======================================================================================================================
# Main Classes
# ======================================================================================================================

class SceneSettings:
    def __init__(self, settings_dict, device, scene_scale_factor=1.):
        scene_settings = settings_dict['scene']

        self.grid_w = scene_settings['grid_w']
        self.grid_h = scene_settings['grid_h']
        self.grid_l = scene_settings['grid_l']

        self.cell_capacity = scene_settings['cell_capacity']
        self.cell_resolution = scene_settings['cell_resolution']

        self.x_min = scene_scale_factor * torch.Tensor(scene_settings['x_min']).to(device)
        self.x_max = scene_scale_factor * torch.Tensor(scene_settings['x_max']).to(device)


class CameraSettings:
    def __init__(self, settings_dict, device, scene_scale_factor=1.):
        camera_settings = settings_dict['camera']
        # todo: Why .long() ?
        self.x_min = scene_scale_factor * torch.Tensor(camera_settings['x_min']).to(device)
        self.x_max = scene_scale_factor * torch.Tensor(camera_settings['x_max']).to(device)
        

        self.pose_w = camera_settings['pose_w']
        self.pose_h = camera_settings['pose_h']
        self.pose_l = camera_settings['pose_l']

        self.pose_n_elev = camera_settings['pose_n_theta']
        self.pose_n_azim = camera_settings['pose_n_azim']

        self.start_positions = torch.Tensor(camera_settings['start_positions']).long().to(device)

        self.contrast_factor = camera_settings['contrast_factor']


class Settings:
    def __init__(self, settings_dict, device, scene_scale_factor=1.):
        self.camera = CameraSettings(settings_dict, device, scene_scale_factor)
        self.scene = SceneSettings(settings_dict, device, scene_scale_factor)


class Camera:
    def __init__(self, x_min, x_max,
                 pose_l, pose_w, pose_h, pose_n_elev, pose_n_azim, n_interpolation_steps,
                 zfar, renderer, device,
                 contrast_factor=1.,
                 gathering_factor=0.05,
                 occupied_pose_data=None,
                 save_dir_path=None,
                 mirrored_scene=False,
                 mirrored_axis=None):
        """
        Main class for camera management in 3D scenes.

        :param x_min: (Tensor) Minimum coordinates for camera poses. Has shape (3,)
        :param x_max: (Tensor) Maximum coordinates for camera poses. Has shape (3,)
        :param pose_l: (int) Number of different camera poses on x-axis
        :param pose_w: (int) Number of different camera poses on y-axis
        :param pose_h: (int) Number of different camera poses on z-axis
        :param pose_n_elev: (int) Number of different elevation angles for camera poses.
        :param pose_n_azim: (int) Number of different azimuth angles for camera poses.
        :param n_interpolation_steps (int) Number of frames to capture from one pose to another.
        :param zfar: (float) Maximum depth value for camera rendering.
        :param renderer: Camera renderer.
        :param device:
        :param save_dir_path: Path to the directory in which frames will be saved.
        :param contrast_factor: (float) Value to adjust output image contrast.
            contrast_factor=1. does not change the result.
        :param gathering_factor: (float) Proportion of points to gather by the sensor in the depth map.
        :param occupied_pose_data: (Dict) Dictionary containing info about occupied poses. Here the occupied poses mean the camera poses are inside the mesh.
        """

        # ----------Pose settings---------------------------------------------------------------------------------------
        self.x_min = 0. + x_min + 3
        self.x_max = 0. + x_max -3
        self.pose_l = pose_l
        self.pose_w = pose_w
        self.pose_h = pose_h
        self.pose_n_elev = pose_n_elev
        self.pose_n_azim = pose_n_azim

        self.n_interpolation_steps = n_interpolation_steps

        if mirrored_scene:
            if mirrored_axis is None:
                raise NameError("Please provide the list of mirrored axis.")
            else:
                for axis in mirrored_axis:
                    self.x_min[..., axis], self.x_max[..., axis] = -self.x_max[..., axis], -self.x_min[..., axis]

        # ----------Rendering settings----------------------------------------------------------------------------------
        self.zfar = zfar
        self.renderer = renderer
        self.image_height = renderer.rasterizer.raster_settings.image_size[0]
        self.image_width = renderer.rasterizer.raster_settings.image_size[1]
        self.contrast_factor = contrast_factor
        self.gathering_factor = gathering_factor

        self.n_frames_captured = 0
        self.save_dir_path = save_dir_path

        self.device = device

        # ----------Current Camera--------------------------------------------------------------------------------------
        self.cam_idx = None
        self.X_cam = None
        self.V_cam = None
        self.fov_camera = None

        # ----------Camera history--------------------------------------------------------------------------------------
        self.cam_idx_history = torch.zeros(0, 5, device=device)
        self.X_cam_history = torch.zeros(0, 3, device=device)
        self.V_cam_history = torch.zeros(0, 2, device=device)

        # ----------Prediction Camera--------------------------------------------------------------------------------------
        self.fov_camera_0 = None

        # ----------Projection tools------------------------------------------------------------------------------------
        x_tab = torch.Tensor([[i for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        y_tab = torch.Tensor([[j for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        self.ndc_x_tab = self.image_width / min(self.image_width,
                                                self.image_height) - (y_tab / (min(self.image_width,
                                                                                   self.image_height) - 1)) * 2
        self.ndc_y_tab = self.image_height / min(self.image_width,
                                                 self.image_height) - (x_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2
        self.min_ndc_x, self.max_ndc_x = self.ndc_x_tab[-1, -1], self.ndc_x_tab[0, 0]
        self.min_ndc_y, self.max_ndc_y = self.ndc_y_tab[-1, -1], self.ndc_y_tab[0, 0]

        # ----------Trajectory management tools-------------------------------------------------------------------------
        # self.pose_space = {}
        # self.pose_history = {}
        # pose_indices = torch.cartesian_prod(torch.arange(0, pose_l),
        #                                     torch.arange(0, pose_w),
        #                                     torch.arange(0, pose_h),
        #                                     torch.arange(0, pose_n_elev),
        #                                     torch.arange(0, pose_n_azim))

        # self.l_step = (self.x_max - self.x_min)[0] / self.pose_l
        # self.w_step = (self.x_max - self.x_min)[1] / self.pose_w
        # self.h_step = (self.x_max - self.x_min)[2] / self.pose_h
        
        # self.l_step = (self.x_max - self.x_min)[0] / (self.pose_l - 1)
        # self.w_step = (self.x_max - self.x_min)[1] / (self.pose_w)
        # self.h_step = (self.x_max - self.x_min)[2] / (self.pose_h - 1)

        self.pose_shift = torch.cartesian_prod(torch.arange(0, 3),
                                               torch.arange(0, 3),
                                               torch.arange(0, 3),
                                               torch.arange(0, 3),
                                               torch.arange(-2, 5)).to(self.device) - 1 # -2, -1, 0, 1, 2
        # For positions, we want the camera to move by exactly 1 unit
        self.pose_shift = self.pose_shift[
            (torch.sum(torch.abs(self.pose_shift[:, :3]), dim=1) == 1).view(-1, 1).expand(-1, 5)].view(-1, 5)
        # For rotations, we want the camera to rotate by at most 1 unit
        # self.pose_shift = self.pose_shift[
        #     (torch.sum(torch.abs(self.pose_shift[:, 4:]), dim=1) <= 4).view(-1, 1).expand(-1, 5)].view(-1, 5)

        # for pose_idx in pose_indices:
        #     i_l, i_w, i_h, i_theta, i_azim = pose_idx[0].item(), pose_idx[1].item(), pose_idx[2].item(), \
        #                                      pose_idx[3].item(), pose_idx[4].item()

        #     pose = torch.Tensor([self.x_min[0] + (1 / 2. + i_l) * self.l_step,
        #                          self.x_min[1] + (1 / 2. + i_w) * self.w_step,
        #                          self.x_min[2] + (1 / 2. + i_h) * self.h_step,
        #                          -90. + 180. * (1 + i_theta) / (self.pose_n_elev + 1),
        #                          360 * i_azim / self.pose_n_azim
        #                          ]).to(device)
        #     self.pose_space[str(pose_idx.numpy().tolist())] = pose
        #     self.pose_history[str(pose_idx.numpy().tolist())] = False

        self.pose_space = {}
        self.pose_history = {}

        # 使用修改后的步长生成pose_indices
        pose_indices = torch.cartesian_prod(torch.arange(0, self.pose_l),
                                            torch.arange(0, self.pose_w),
                                            torch.arange(0, self.pose_h),
                                            torch.arange(0, self.pose_n_elev),
                                            torch.arange(0, self.pose_n_azim)).to(device)

        self.l_step = self.h_step = 3
        # t1 = time.time()
        # for pose_idx in pose_indices:
        #     i_l, i_w, i_h, i_theta, i_azim = pose_idx
        #     pose = torch.Tensor([self.x_min[0] + i_l * self.l_step,
        #                         self.x_min[1] + 0.7,
        #                         self.x_min[2] + i_h * self.h_step,
        #                         -90. + 180. * (1 + i_theta) / (self.pose_n_elev + 1),
        #                         360 * i_azim / self.pose_n_azim
        #                         ]).to(device)
        #     self.pose_space[str(pose_idx.numpy().tolist())] = pose
        #     self.pose_history[str(pose_idx.numpy().tolist())] = False
        # t2 = time.time()
        # print(t2-t1)
        # print(self.pose_space)
        poses = torch.zeros(pose_indices.shape[0], 5, device=device)
        poses[:, 0] = x_min[0] + pose_indices[:, 0] * self.l_step
        poses[:, 1] = x_min[1] + 3.3  # 假设y坐标固定为0.7
        poses[:, 2] = x_min[2] + pose_indices[:, 2] * self.h_step
        poses[:, 3] = -90. + 180. * (1 + pose_indices[:, 3]) / (self.pose_n_elev + 1)
        poses[:, 4] = 360. * pose_indices[:, 4] / self.pose_n_azim

        # 生成字符串键
        keys = [str(list(item)) for item in pose_indices.cpu().numpy()]
        for key, pose in zip(keys, poses):
            self.pose_space[key] = pose
            self.pose_history[key] = False

        # ----------Occupied pose management----------------------------------------------------------------------------
        self.use_occupied_pose = False
        # if occupied_pose_data is not None:
        #     self.use_occupied_pose = True
        #     pose_is_occupied = {}
        #     for i in range(len(occupied_pose_data['X_idx'])):
        #         is_occupied = occupied_pose_data['occupied'][i]

        #         X_idx = 0. + occupied_pose_data['X_idx'][i]
        #         if mirrored_scene:
        #             if mirrored_axis is None:
        #                 raise NameError("Please provide the list of mirrored axis.")
        #             else:
        #                 for axis in mirrored_axis:
        #                     if axis == 0:
        #                         bound = self.pose_l - 1
        #                     elif axis == 1:
        #                         bound = self.pose_w - 1
        #                     elif axis == 2:
        #                         bound = self.pose_h - 1
        #                     else:
        #                         raise NameError("Wrong value for axis to be mirrored. "
        #                                         "Please choose an integer in [0, 1, 2].")
        #                     X_idx[..., axis] = bound - X_idx[..., axis]

        #         X_key = str(X_idx.long().cpu().numpy().tolist())

        #         pose_is_occupied[X_key] = is_occupied.item()

        #     self.pose_is_occupied = pose_is_occupied

    def initialize_camera(self, start_cam_idx):
        """
        Initialize a camera with a start index.

        :param start_cam_idx: (List, Array or Tensor)
        :return: None
        """
        self.update_camera(start_cam_idx)
        self.fov_camera_0 = FoVPerspectiveCameras(R=self.fov_camera.R, T=self.fov_camera.T, zfar=self.zfar,
                                                  device=self.device)
        
    def generate_new_splited_dict(self):
        '''
        Split the dict to two tensors

        :param: self.pose_space is the possible poses of whole scene
        '''
        new_d = {}
        for k, v in self.pose_space.items():
            pose_is_occupied = self.check_if_pose_is_occupied(k, input_type='key')
            if not pose_is_occupied:
                first_tensor = v[:3]
                second_tensor = v[2:]
                new_d[k] = (first_tensor, second_tensor)
        return new_d
    
    def generate_position_list(self, new_d):
        first_tensors = torch.vstack([v[0].flatten() for v in new_d.values()])
        unique_first_tensors = torch.unique(first_tensors, dim=0)
        return unique_first_tensors

    def sample_valid_poses_in_a_certain_range(self, mesh, proxy_scene, sampled_range_value, current_camera_position, camera_given_min_range, camera_given_max_range):
        '''
        Reuturn a list of valid sampled camera poses in a certain range.

        :param range_value: Define the certain range that we will sample our camera poses
        '''

        def generate_positions_with_combinations(current_camera_position, sampled_range_value, camera_given_min_range, camera_given_max_range):
            all_positions = [[current_camera_position[0]+dx, current_camera_position[1]+dy, current_camera_position[2]+dz] 
                            for dx in range(-sampled_range_value, sampled_range_value+1)
                            for dy in range(-sampled_range_value, sampled_range_value+1)
                            for dz in range(-sampled_range_value, sampled_range_value+1)
                            if abs(dx) + abs(dy) + abs(dz) <= sampled_range_value and 
                            all(camera_given_min_range[i] <= current_camera_position[i]+[dx, dy, dz][i] <= camera_given_max_range[i] for i in range(3))]

            # Generate all possible combinations of suffixes
            suffixes = [[i, j] for i in range(5) for j in range(10)]

            # Add all possible suffixes for each position
            extended_positions = [pos + suffix for pos in all_positions for suffix in suffixes if pos != current_camera_position]
            
            return extended_positions
        
        generated_camera_pose_idx_list = generate_positions_with_combinations(current_camera_position, sampled_range_value, camera_given_min_range, camera_given_max_range)
        sample_poses_indices = random.sample(generated_camera_pose_idx_list, len(generated_camera_pose_idx_list) // 5)
       
        sampled_poses_params_list = []
        for indices in sample_poses_indices:
            # Check if pose is occupied
            indices = str(indices)
            pose_is_occupied = self.check_if_pose_is_occupied(indices, input_type='key')

            if not pose_is_occupied:
                # Check if pose has an empty field of view
                random_pose = self.pose_space[indices]
                X_cam, V_cam, fov_camera = self.get_camera_parameters_from_pose(random_pose) # Important
                empty_fov = self.is_fov_empty(mesh, fov_camera)

                if not empty_fov:
                # Check if pose is oriented toward bounding box
                    fov_proxy_points = self.get_points_in_fov(pts=proxy_scene.proxy_points, return_mask=False,
                                                            fov_camera=fov_camera, fov_range=5 * self.zfar)
                    no_proxy_in_fov = fov_proxy_points.shape[0] <= 0

                    if not no_proxy_in_fov:
                        params_list = [random_pose, X_cam, fov_camera, indices]
                        sampled_poses_params_list.append(params_list)

        return sampled_poses_params_list

    def sample_valid_poses_in_2d_space(self, mesh, proxy_scene, num_samples=200):
        '''
        By Shiyao
        Return a list of valid sampled poses in whole scene. 
        Every item in the list will be like: [camera_pose, X_cam, fov_camera, idx]

        :param mesh
        :param proxy_scene
        '''

        # Sample num_samples random poses
        sampled_poses_indices = np.random.choice(list(self.pose_space.keys()), num_samples)
        current_pose = self.cam_idx.tolist()
        # Used for storing the params of valid sampled poses.
        sampled_poses_params_list = []
        stacked_poses_list = []
        for indices in sampled_poses_indices:
            # Check if pose is occupied
            if indices == current_pose:
                continue
            pose_is_occupied = self.check_if_pose_is_occupied(indices, input_type='key')

            if not pose_is_occupied:
                # Check if pose has an empty field of view
                random_pose = self.pose_space[indices]
                X_cam, V_cam, fov_camera = self.get_camera_parameters_from_pose(random_pose) # Important
                empty_fov = self.is_fov_empty(mesh, fov_camera)

                if not empty_fov:
                # Check if pose is oriented toward bounding box
                    fov_proxy_points = self.get_points_in_fov(pts=proxy_scene.proxy_points, return_mask=False,
                                                            fov_camera=fov_camera, fov_range=5 * self.zfar)
                    no_proxy_in_fov = fov_proxy_points.shape[0] <= 0

                    if not no_proxy_in_fov:
                        params_list = [random_pose, X_cam, indices]
                        sampled_poses_params_list.append(params_list)
                        stacked_poses_list.append(random_pose)
        stacked_poses = torch.stack(stacked_poses_list)

        return stacked_poses, sampled_poses_params_list
    
    def sample_valid_poses_in_space(self, mesh, proxy_scene, num_samples=200):
        '''
        By Shiyao
        Return a list of valid sampled poses in whole scene. 
        Every item in the list will be like: [camera_pose, X_cam, fov_camera, idx]

        :param mesh
        :param proxy_scene
        '''

        # Sample num_samples random poses
        sampled_poses_indices = np.random.choice(list(self.pose_space.keys()), num_samples)
        # Used for storing the params of valid sampled poses.
        sampled_poses_params_list = []
        for indices in sampled_poses_indices:
            # Check if pose is occupied
            pose_is_occupied = self.check_if_pose_is_occupied(indices, input_type='key')

            if not pose_is_occupied:
                # Check if pose has an empty field of view
                random_pose = self.pose_space[indices]
                X_cam, V_cam, fov_camera = self.get_camera_parameters_from_pose(random_pose) # Important
                empty_fov = self.is_fov_empty(mesh, fov_camera)

                if not empty_fov:
                # Check if pose is oriented toward bounding box
                    fov_proxy_points = self.get_points_in_fov(pts=proxy_scene.proxy_points, return_mask=False,
                                                            fov_camera=fov_camera, fov_range=5 * self.zfar)
                    no_proxy_in_fov = fov_proxy_points.shape[0] <= 0

                    if not no_proxy_in_fov:
                        params_list = [random_pose, X_cam, fov_camera, indices]
                        sampled_poses_params_list.append(params_list)

        return sampled_poses_params_list
                
    def get_random_valid_pose(self, mesh, proxy_scene):
        """
        Returns the idx of a valid camera pose.
        A pose is considered invalid iff it is either occupied (inside an object) or has an empty field of view.

        :param mesh: (Mesh) Mesh of the object to reconstruct in the scene.
        :param proxy_scene (Scene)
        :return:
        """
        random_pose_key = None
        is_not_valid = True

        while is_not_valid:
            # Draw a random pose
            random_pose_key = np.random.choice(list(self.pose_space.keys()))

            # Check if pose is occupied
            pose_is_occupied = self.check_if_pose_is_occupied(random_pose_key, input_type='key')

            # Check if pose has an empty field of view
            random_pose = self.pose_space[random_pose_key]
            X_cam, V_cam, fov_camera = self.get_camera_parameters_from_pose(random_pose)
            empty_fov = self.is_fov_empty(mesh, fov_camera)

            # Check if pose is oriented toward bounding box
            fov_proxy_points = self.get_points_in_fov(pts=proxy_scene.proxy_points, return_mask=False,
                                                      fov_camera=fov_camera, fov_range=5 * self.zfar)
            no_proxy_in_fov = fov_proxy_points.shape[0] <= 0

            is_not_valid = empty_fov or pose_is_occupied or no_proxy_in_fov

        valid_pose_idx = self.get_idx_from_key(random_pose_key)
        return valid_pose_idx
    
    def get_neighboring_poses(self, pose_idx=None):
        """
        Return neighboring poses' indices of provided camera pose index.
        if pose_idx is None, return neighbors of current camera pose.

        :param pose_idx: (List, Array or Tensor)
        :return: (Tensor)
        """
        if pose_idx is None:
            pose_idx = self.cam_idx

        res = pose_idx + self.pose_shift
        res[..., :4][res[..., :4] < 0] = 0
        res[..., 0][res[..., 0] >= self.pose_l] = self.pose_l - 1
        # Fixing the y axis value by setting it back to the original
        res[..., 1] = pose_idx[..., 1]
        res[..., 2][res[..., 2] >= self.pose_h] = self.pose_h - 1
        res[..., 3] = pose_idx[..., 3]


        res[..., 4] = res[..., 4] % self.pose_n_azim

        # We remove neighbor poses where the camera does not translate
        res = res[(torch.sum(torch.abs((res - pose_idx)[..., :3]), dim=-1) > 0.)]

        return torch.unique(res, dim=0)
        
    
    def get_neighboring_poses_single_location(self, pose_idx=None, next_idx=None):
        # todo: Use torch.clamp rather than inequalities
        """
        Return neighboring poses' indices of provided camera pose index.
        if pose_idx is None, return neighbors of current camera pose.

        :param pose_idx: (List, Array or Tensor)
        :return: (Tensor)
        """
        # TO CHANGE! Remove the poses where the camera makes no translation move.
        if pose_idx is None:
            pose_idx = self.cam_idx

        res = pose_idx + self.pose_shift
        res[..., :4][res[..., :4] < 0] = 0
        res[..., 0][res[..., 0] >= self.pose_l] = self.pose_l - 1
        res[..., 1][res[..., 1] >= self.pose_w] = self.pose_w - 1
        res[..., 2][res[..., 2] >= self.pose_h] = self.pose_h - 1
        res[..., 3][res[..., 3] >= self.pose_n_elev] = self.pose_n_elev - 1

        res[..., 4] = res[..., 4] % self.pose_n_azim

        # We remove neighbor poses where camera does not translate
        res = res[torch.all(torch.isclose(res[..., :3], next_idx, atol=1e-5), dim=-1)]

        return torch.unique(res, dim=0)

    def get_valid_neighbors(self, neighbor_indices, mesh):
        """
        Returns the valid, non visited camera poses among the neighbors.
        If such a pose does not exit, returns the already visited (and necessarily valid) poses.

        :param neighbor_indices: (Tensor) Neighbors indices tensor with shape (n_neighbors, 5).
        :param mesh:
        :return: (Tensor) Has shape (n_result, 5)
        """
        new_valid_neighbors = torch.zeros(0, neighbor_indices.shape[1], device=self.device).long()
        visited_neighbors = torch.zeros(0, neighbor_indices.shape[1], device=self.device).long()

        for i in range(len(neighbor_indices)):
            neighbor_idx = neighbor_indices[i]

            # We check if it has been visited
            _, is_visited = self.get_pose_from_idx(neighbor_idx)
            if is_visited:
                visited_neighbors = torch.vstack((visited_neighbors, neighbor_idx))
            else:
                is_valid = self.check_if_pose_is_valid(mesh=mesh, pose=neighbor_idx, input_type='idx')
                if is_valid:

                    new_valid_neighbors = torch.vstack((new_valid_neighbors, neighbor_idx))

        if new_valid_neighbors.shape[0] > 0:
            return new_valid_neighbors
        else:
            return visited_neighbors

    def get_idx_from_key(self, pose_key):
        """
        Return, for any pose key from pose_space dictionary, the corresponding index tensor.

        :param pose_key: (str)
        :return: (Tensor)
        """
        pose_idx = torch.Tensor([int(c) for c in pose_key[1:-1].split(',')]).long().to(self.device)
        return pose_idx

    def get_key_from_idx(self, pose_idx):
        """
        Return, for any pose index, the corresponding key in pose dictionary self.pose_space.

        :param pose_idx: (List, Array or Tensor)
        :return: (str)
        """
        key = str(pose_idx.cpu().numpy().tolist())
        return key

    def get_pose_from_idx(self, pose_idx):
        """
        Return, for any pose index, the corresponding 5D pose.

        :param pose_idx: (List, Array or Tensor)
        :return: (Tensor, bool)
        """
        key = self.get_key_from_idx(pose_idx)
        pose = self.pose_space[key]
        visited = self.pose_history[key]
        return pose, visited

    def update_camera(self, new_cam_index, interpolation_step=None):
        """
        Update current camera parameters given a new camera pose index.

        :param new_cam_index: (List, Array or Tensor)
        :param interpolation_step: (int) Should be in the closed interval [1, n_interpolation_steps]
        :return: None
        """
        if interpolation_step is None:
            interpolation_step = self.n_interpolation_steps

        elif interpolation_step > self.n_interpolation_steps:
            raise NameError("interpolation_step is too large!"
                            "It should be in the closed interval [1, n_interpolation_steps].")

        if interpolation_step == self.n_interpolation_steps:
            self.cam_idx = new_cam_index
            self.cam_idx_history = torch.vstack((self.cam_idx_history, new_cam_index))
            self.pose_history[self.get_key_from_idx(new_cam_index)] = True

        old_pose, _ = self.get_pose_from_idx(self.cam_idx)
        new_pose, _ = self.get_pose_from_idx(new_cam_index)
        old_X, old_V = old_pose[:3].view(1, 3), old_pose[3:].view(1, 2)
        new_X, new_V = new_pose[:3].view(1, 3), new_pose[3:].view(1, 2)

        if interpolation_step == self.n_interpolation_steps:
            offset_azim = 0.
        elif (self.cam_idx[..., -1] == 0) and (new_cam_index[..., -1] == self.pose_n_azim-1):
            offset_azim = -360.
        elif (self.cam_idx[..., -1] == self.pose_n_azim - 1) and (new_cam_index[..., -1] == 0):
            offset_azim = 360.
        else:
            offset_azim = 0.

        self.X_cam = old_X + (new_X - old_X) * interpolation_step / self.n_interpolation_steps
        self.V_cam = old_V + (new_V - old_V) * interpolation_step / self.n_interpolation_steps
        self.V_cam[..., -1] = self.V_cam[..., -1] + offset_azim * interpolation_step / self.n_interpolation_steps

        self.X_cam_history = torch.vstack((self.X_cam_history, self.X_cam))
        self.V_cam_history = torch.vstack((self.V_cam_history, self.V_cam))

        R_cam, T_cam = get_camera_RT(self.X_cam, self.V_cam)
        self.fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=self.zfar, device=self.device)

    def get_camera_parameters_from_pose(self, cam_pose):
        """
        Return camera parameters corresponding to the provided camera pose.

        :param cam_pose: (Tensor) Full camera pose tensor with shape (1, 5)
        :return: (Tensor, Tensor, FoVPerspectiveCameras) Position tensor X_cam has shape (1, 3)
            and elevation/azimuth angles tensor V_cam has shape (1, 2)
        """
        X_cam = cam_pose[:3].view(1, 3)
        V_cam = cam_pose[3:].view(1, 2)

        R_cam, T_cam = get_camera_RT(X_cam, V_cam)
        fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=self.zfar, device=self.device)

        return X_cam, V_cam, fov_camera

    def get_fov_camera_from_RT(self, R_cam, T_cam):
        """
        Return the FoVPerspectiveCameras object corresponding to camera poses R_cam and T_cam.

        :param R_cam: (Tensor) Position tensor with shape (n_camera, 3, 3)
        :param T_cam: (Tensor) Elevation and azimuth tensor with shape (n_camera, 3)
        :return: (FoVPerspectiveCameras)
        """
        fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=self.zfar, device=self.device)
        return fov_camera

    def get_fov_camera_from_XV(self, X_cam, V_cam):
        """
        Return the FoVPerspectiveCameras object corresponding to camera poses X_cam and V_cam.

        :param X_cam: (Tensor) Position tensor with shape (n_camera, 3)
        :param V_cam: (Tensor) Elevation and azimuth tensor with shape (n_camera, 2)
        :return: (FoVPerspectiveCameras)
        """
        R_cam, T_cam = get_camera_RT(X_cam, V_cam)
        return self.get_fov_camera_from_RT(R_cam, T_cam)

    def is_fov_empty(self, mesh, fov_camera=None):
        """
        Return a boolean that indicates if the field of view of fov_camera is empty or not.
        'Empty' means that no vertex of the mesh is inside the field of view of fov_camera.
        Can be confusing for meshes with very large triangles.

        :param fov_camera: (FoVPerspectiveCameras)
        :param mesh: (Mesh)
        :return: (bool)
        """
        if fov_camera is None:
            fov_camera = self.fov_camera

        fov_verts = self.get_points_in_fov(pts=mesh.verts_list()[0], return_mask=False,
                                           fov_camera=fov_camera, fov_range=5*self.zfar)
        empty_fov = fov_verts.shape[0] <= 0
        return empty_fov

    def check_if_pose_is_occupied(self, pose, input_type='idx'):
        """
        Return a boolean indicating if the pose is occupied or not in order to avoid collisions.

        :param pose:
        :param input_type: (str) Indicates which type of input is provided.
        Can be 'idx', 'key'.
        :return: (bool)
        """

        if self.use_occupied_pose:
            if input_type == 'key':
                pose_key = pose
            elif input_type == 'idx':
                pose_key = self.get_key_from_idx(pose)
            else:
                raise NameError("Wrong input_type argument. Please select between 'idx' and 'key'.")

            X_key = '[' + ','.join(pose_key[1:-1].split(',')[:3]) + ']'
            is_occupied = self.pose_is_occupied[X_key]
        else:
            is_occupied = False

        return False #is_occupied

    def check_if_pose_is_valid(self, mesh, pose, input_type='idx'):
        """
        Return True if the camera pose is valid.
        A pose is considered invalid iff it is either occupied or has an empty field of view.

        :param mesh:
        :param pose:
        :param input_type: (str) Input type for pose argument. Can be 'idx' or 'key'.
        :return:
        """

        # Check if pose is occupied
        # pose_is_occupied = self.check_if_pose_is_occupied(pose, input_type=input_type)
        pose_is_occupied = False
        # Check if pose has an empty field of view
        if input_type == 'idx':
            pose_idx = pose
        elif input_type == 'key':
            pose_idx = self.get_idx_from_key(pose)
        else:
            raise NameError("Argument input_type is invalid. Please choose between 'idx' and 'key'.")
        random_pose, _ = self.get_pose_from_idx(pose_idx)
        X_cam, V_cam, fov_camera = self.get_camera_parameters_from_pose(random_pose)
        empty_fov = self.is_fov_empty(mesh, fov_camera)

        is_valid = not(empty_fov or pose_is_occupied)
        return is_valid

    def capture_image(self, mesh, fov_camera=None,
                      save_frame=True, dir_path=None):
        """
        Render an image of the scene from the current camera.

        :param mesh:
        :param fov_camera: (FoVPerpsectiveCamera) if None, use the current camera.
        :param save_frame: (bool) If True, save frame to the corresponding directory.
        :param dir_path: (str) Path to the directory in which the frame will be saved.
            If None, default path provided during initialization will be used.
        :return: (2-Tuple of Tensors)
        """
        if fov_camera is None:
            fov_camera = self.fov_camera

        with torch.no_grad():
            images, fragments = self.renderer(mesh, cameras=fov_camera)
            images = adjust_contrast(transpose_channels(images[..., :3], True), self.contrast_factor)
            images = transpose_channels(images, False)
            depth = fragments.zbuf

        dir_path_provided = (self.save_dir_path is not None) or (dir_path is not None)
        
        if save_frame and dir_path_provided:
            if dir_path is None:
                dir_path = self.save_dir_path

            # Save image and data
            mask = depth > -1

            img_dir = {}
            img_dir['rgb'] = images
            img_dir['zbuf'] = depth
            img_dir['mask'] = mask
            img_dir['R'] = fov_camera.R
            img_dir['T'] = fov_camera.T
            img_dir['zfar'] = self.zfar
            frame_name = str(self.n_frames_captured) + ".pt"
            frame_save_path = os.path.join(dir_path, frame_name)
            torch.save(img_dir, frame_save_path)

            self.n_frames_captured += 1

        return images, depth

    def project_depth_in_3D(self, depth, fov_cameras=None):
        """
        For a set of depth maps and associated cameras, project the DM back to 3D.

        :param depth: (Tensor) Depth map tensor with shape (batch_size, height, width, 1)
        :param fov_cameras:
        :return:
        """
        batch_size = depth.shape[0]

        ndc_points = torch.cat((self.ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                self.ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                depth.view(batch_size, -1, 1)),
                               dim=-1
                               ).view(batch_size, self.image_height * self.image_width, 3)

        # We reproject the points in world space
        if fov_cameras is None:
            fov_cameras = self.fov_camera

        all_world_points = fov_cameras.unproject_points(ndc_points, scaled_depth_input=False)
        return all_world_points

    def compute_partial_point_cloud(self, depth, mask, images=None, fov_cameras=None,
                                    gathering_factor=None, fov_range=None):
        """
        Compute the partial point cloud from a single camera.

        :param depth: (Tensor) Depth map tensor with shape (1, height, width, 1)
        :param mask: (Tensor) Mask tensor with shape (1, height, width, 1)
        :param images: (Tensor) Image tensor with shape (1, height, width, 3)
        :param fov_cameras:
        :return:
        """
        if fov_range is None:
            points_mask = mask.view(1, -1)
        else:
            points_mask = mask.view(1, -1) * (depth < fov_range).view(1, -1)

        # We reproject the points in world space
        all_world_points = self.project_depth_in_3D(depth, fov_cameras=fov_cameras)

        # We filter the points with the mask
        world_points = all_world_points[points_mask]

        # We keep only a fraction of points
        if gathering_factor is None:
            gathering_factor = self.gathering_factor
        n_points = int(len(world_points) * gathering_factor)  # -1, 2048
        points_indices = torch.randperm(len(world_points))[:n_points]
        world_points = world_points[points_indices]

        if images is None:
            return world_points

        else:
            all_points_color = 0. + images.view(1, -1, 3)
            points_color = all_points_color[points_mask]
            points_color = points_color[points_indices]
            return world_points, points_color

    def get_points_in_fov(self, pts, return_mask=False, fov_camera=None, fov_range=None):
        """
        Return the points in pts that are inside the field of view of fov_camera,
        in a range lower or equal to fov_range.

        :param pts: (Tensor) Points tensor with shape (n_point, 3)
        :param return_mask: (bool) If True, return the boolean mask to compute the result from pts.
        :param fov_camera:
        :param fov_range (float) If None, fov_range is equal to 1.1 * self.zfar
        :return:
        """
        if fov_camera is None:
            fov_camera = self.fov_camera
            camera_center = self.X_cam
        else:
            camera_center = fov_camera.get_camera_center()

        scene_projections = fov_camera.get_full_projection_transform().transform_points(pts)
        scene_view = fov_camera.get_world_to_view_transform().transform_points(pts)

        projection_mask = (scene_projections[:, 0] >= self.min_ndc_x) * \
                          (scene_projections[:, 0] <= self.max_ndc_x) * \
                          (scene_projections[:, 1] >= self.min_ndc_y) * \
                          (scene_projections[:, 1] <= self.max_ndc_y) * \
                          (scene_view[:, 2] > 0.)

        if fov_range is None:
            fov_mask = projection_mask
        else:
            range_mask = torch.linalg.norm(pts - camera_center, dim=-1) < fov_range
            fov_mask = range_mask * projection_mask

        if return_mask:
            return pts[fov_mask], fov_mask
        else:
            return pts[fov_mask]

    def get_points_zbuf(self, pts, fov_camera=None):
        """
        Return zbuf, i.e. z coordinates of points 'pts' in the view space of camera 'fov_camera'.

        :param pts: (Tensor) 3D points tensor with shape (n_points, 3).
        :param fov_camera:
        :return: (Tensor) Has shape (n_camera, n_points, 1)
        """
        if fov_camera is None:
            fov_camera = self.fov_camera

        pts_zbuf = fov_camera.get_world_to_view_transform().transform_points(pts)[..., 2:]
        return pts_zbuf

    def get_signed_distance_to_depth_maps(self, pts, depth_maps, mask, fov_camera=None):
        """
        Return the signed distances to surfaces delimited by each depth map.

        :param pts: (Tensor) 3D points tensor with shape (n_points, 3)
        :param depth_maps: (Tensor) Depth maps tensor with shape (n_depth, height, width, 1)
        :param mask: (Tensor) Mask tensor with shape (n_depth, height, width, 1)
        :param fov_camera:
        :return: (Tensor) Signed distance tensor with shape (n_depth, n_points, 1).
            If positive, the point is located "behind" the depth map.
            If negative, the point is located "before" the depth map.
        """
        n_depths = depth_maps.shape[0]
        if fov_camera is None:
            fov_camera = self.fov_camera
            if n_depths > 1:
                raise NameError("Too many depth maps provided for current camera; depth_maps should have shape (1, ...)"
                                "If you want to simultaneously process depth maps for multiple cameras, "
                                "please provide the corresponding fov_camera argument.")
        else:
            if n_depths != fov_camera.R.shape[0]:
                raise NameError("Number of cameras should be the same as number of depths.")

        # Depth of points in the scene
        pts_zbuf = self.get_points_zbuf(pts, fov_camera=fov_camera)

        # Corresponding depth in depth maps
        depths = 0. + depth_maps
        depths[~mask.view(n_depths, self.image_height, self.image_width)] = 1.1 * self.zfar
        depths = transpose_channels(depths, channel_is_at_the_end=True)

        pts_projections = fov_camera.get_full_projection_transform().transform_points(pts)

        factor = -1 * min(self.image_height, self.image_width)
        # todo: Parallelize these two lines with a tensor [image_width, image_height]
        pts_projections[..., 0] = factor / self.image_width * pts_projections[..., 0]
        pts_projections[..., 1] = factor / self.image_height * pts_projections[..., 1]

        pts_projections = pts_projections[..., :2]
        pts_projections = pts_projections.view(n_depths, -1, 1,
                                               2)  # scene_projections.view(batch_size, height, width, 2)

        map_zbuf = torch.nn.functional.grid_sample(input=depths,
                                                   grid=pts_projections,
                                                   mode='bilinear',
                                                   padding_mode='border'  # 'reflection', 'zeros'
                                                   )
        map_zbuf = transpose_channels(map_zbuf, channel_is_at_the_end=False).view(n_depths, -1, 1)

        return pts_zbuf - map_zbuf


class Cell:
    def __init__(self, center, l, w, h, capacity, resolution, device, feature_dim=0):
        self.center = center
        self.l = l
        self.w = w
        self.h = h

        self.x_min = center - torch.Tensor([[l / 2., w / 2., h / 2.]]).to(center.get_device())
        self.x_max = center + torch.Tensor([[l / 2., w / 2., h / 2.]]).to(center.get_device())

        if resolution is None:
            if capacity is None:
                raise NameError("Please choose a capacity or a resolution.")
            else:
                self.capacity = capacity
                a1 = (l * torch.sqrt(w ** 2 + h ** 2)).item()
                a2 = (w * torch.sqrt(h ** 2 + l ** 2)).item()
                a3 = (h * torch.sqrt(l ** 2 + w ** 2)).item()
                area = max(a1, a2, a3)

                area_per_point = area / self.capacity
                radius = np.sqrt(area_per_point / np.pi)

                self.resolution = 2 * radius

        elif capacity is None:
            self.resolution = resolution
            a1 = (l * torch.sqrt(w ** 2 + h ** 2)).item()
            a2 = (w * torch.sqrt(h ** 2 + l ** 2)).item()
            a3 = (h * torch.sqrt(l ** 2 + w ** 2)).item()
            area = max(a1, a2, a3)

            radius = resolution / 2.
            area_per_point = np.pi * radius**2
            self.capacity = int(area // area_per_point)

        else:
            self.resolution = resolution
            self.capacity = capacity

        self.device = device

        self.cell_pts = torch.zeros(0, 3).to(device)
        self.use_feature = feature_dim > 0
        if self.use_feature:
            self.feature_dim = feature_dim
            self.cell_features = torch.zeros(0, feature_dim).to(device)

    def fill(self, pts, features=None, n_point_min=0):
        mask = torch.max(pts - self.x_max, dim=-1)[0] < 0.
        pts_to_add = pts[mask]
        # tolerance = self.resolution / 3 # 根据你的具体需求调整此值
        
        # # 扩展边界以包含靠近边界的点
        # expanded_x_min = self.x_min - tolerance
        # expanded_x_max = self.x_max + tolerance
        
        # # 首先，检查哪些点在扩展后的边界内
        # mask_max = torch.max(pts - expanded_x_max, dim=-1)[0] < 0
        # mask_min = torch.min(pts - expanded_x_min, dim=-1)[0] > 0
        # mask = mask_max & mask_min
        # pts_to_add = pts[mask]
        if self.use_feature and features is not None:
            features_to_add = features[mask]

        if pts_to_add.shape[0] > 0:
            mask = torch.min(pts_to_add - self.x_min, dim=-1)[0] > 0.
            pts_to_add = pts_to_add[mask]
            if self.use_feature and features is not None:
                features_to_add = features_to_add[mask]

            if pts_to_add.shape[0] > n_point_min:

                if self.cell_pts.shape[0] > 0:
                    dists = torch.min(torch.cdist(pts_to_add.double(), self.cell_pts.double(), p=2.0), dim=-1)[0]
                    mask = dists > self.resolution
                    pts_to_add = pts_to_add[mask]
                    if self.use_feature and features is not None:
                        features_to_add = features_to_add[mask]

                self.cell_pts = torch.vstack((self.cell_pts, pts_to_add))
                mask = torch.randperm(len(self.cell_pts))[:self.capacity]
                self.cell_pts = self.cell_pts[mask]
                if self.use_feature and features is not None:
                    self.cell_features = torch.vstack((self.cell_features, features_to_add))
                    self.cell_features = self.cell_features[mask]

    def empty(self):
        self.cell_pts = torch.zeros(0, 3).to(self.device)
        if self.use_feature:
            self.cell_features = torch.zeros(0, self.feature_dim).to(self.device)

    def is_empty(self):
        return self.cell_pts.shape[0] == 0

class Scene:
    def __init__(self, x_min, x_max,
                 grid_l, grid_w, grid_h, cell_capacity, cell_resolution, n_proxy_points,
                 device,
                 view_state_n_elev=7, view_state_n_azim=2*7, feature_dim=0,
                 mirrored_scene=False, score_threshold=1.,
                 mirrored_axis=None):

        # ----------General Parameters----------------------------------------------------------------------------------
        self.grid_l = grid_l
        self.grid_w = grid_w
        self.grid_h = grid_h

        self.x_min = 0. + x_min  # torch.min(verts, dim=0)[0] * 0.99
        self.x_max = 0. + x_max  # torch.max(verts, dim=0)[0] * 1.01

        self.mirrored_scene = mirrored_scene
        self.mirrored_axis = mirrored_axis
        if mirrored_scene:
            if mirrored_axis is None:
                raise NameError("Please provide the list of mirrored axis.")
            else:
                for axis in mirrored_axis:
                    self.x_min[..., axis], self.x_max[..., axis] = -self.x_max[..., axis], -self.x_min[..., axis]

        self.cells = {}
        cell_indices = torch.cartesian_prod(torch.arange(0, grid_l),
                                            torch.arange(0, grid_w),
                                            torch.arange(0, grid_h))

        self.l = (self.x_max - self.x_min)[0] / self.grid_l
        self.w = (self.x_max - self.x_min)[1] / self.grid_w
        self.h = (self.x_max - self.x_min)[2] / self.grid_h

        for cell_idx in cell_indices:
            i_l, i_w, i_h = cell_idx[0].item(), cell_idx[1].item(), cell_idx[2].item()

            center = torch.Tensor([self.x_min[0] + (1 / 2. + i_l) * self.l,
                                   self.x_min[1] + (1 / 2. + i_w) * self.w,
                                   self.x_min[2] + (1 / 2. + i_h) * self.h]).to(device)

            self.cells[str(cell_idx.numpy().tolist())] = Cell(center=center, l=self.l, w=self.w, h=self.h,
                                                              capacity=cell_capacity, resolution=cell_resolution,
                                                              device=device, feature_dim=feature_dim)
            if cell_resolution is None:
                cell_resolution = self.cells[str(cell_idx.numpy().tolist())].resolution
            if cell_capacity is None:
                cell_capacity = self.cells[str(cell_idx.numpy().tolist())].capacity

        self.cell_resolution = cell_resolution
        self.cell_capacity = cell_capacity
        self.feature_dim = feature_dim
        self.device = device

        # ----------Proxy Points Management-----------------------------------------------------------------------------
        # To use proxy points, please call the method initialize_proxy_points().

        # Proxy Points
        self.n_proxy_points = n_proxy_points
        self.proxy_points = None

        # Predicted occupancy probability of Proxy Points
        self.proxy_proba = None

        # Supervision Occupancy for Proxy Points, computed from depth maps
        self.proxy_supervision_occ = None

        # View State vectors for Proxy Points
        self.view_states = None
        self.view_state_n_elev = view_state_n_elev
        self.view_state_n_azim = view_state_n_azim
        self.n_view_state_cameras = view_state_n_azim * view_state_n_elev

        # View score data for Proxy Points. Used and updated when computing pseudo-GT supervision occupancy
        self.proxy_n_inside_fov = None  # For each point p, number of images for which p is in th fov
        self.proxy_n_behind_depth = None  # For each point p, number of images for which p is behind the depth map
        self.score_threshold = score_threshold

        # Out Of Field values for Proxy Points
        self.out_of_field = None

        # Typical distance between Proxy Points
        n_proxy_points_per_cell = n_proxy_points / (grid_l * grid_h * grid_w)
        cell_volume = self.l * self.w * self.h
        volume_per_proxy_point = cell_volume / n_proxy_points_per_cell
        proxy_radius = np.power(3 * volume_per_proxy_point.item() / (4 * np.pi), 1./3.)
        self.distance_between_proxy_points = 2 * proxy_radius

    def get_pts_in_bounding_box(self, pts, return_mask=True):
        """
        Return points located inside the bounding box of the scene.

        :param pts: (Tensor) 3D Points tensor with shape (n_points, 3)
        :param return_mask: (bool) If True, return the mask to compute the result from pts.
        :return: (Tensor) or (Tensor, Tensor)
        """
        pts_mask = (pts >= self.x_min) * (pts <= self.x_max)
        pts_mask = pts_mask.prod(dim=-1).bool()
        if return_mask:
            return pts[pts_mask], pts_mask
        else:
            return pts[pts_mask]

    # todo: Do floor_divide for each axis at the same time, by creating a 'steps' tensor [l, w, h].
    #  Then, use torch.clamp rather than inequalities.
    def get_cells_for_each_pt(self, pts):
        i_l = floor_divide((pts - self.x_min)[:, 0:1], self.l)
        i_w = floor_divide((pts - self.x_min)[:, 1:2], self.w)
        i_h = floor_divide((pts - self.x_min)[:, 2:3], self.h)

        i_l[i_l >= self.grid_l] = self.grid_l - 1
        i_w[i_w >= self.grid_w] = self.grid_w - 1
        i_h[i_h >= self.grid_h] = self.grid_h - 1

        res = torch.hstack((i_l, i_w, i_h)).long()
        res[res < 0] = 0

        return res

    def get_englobing_cells(self, pts, list=False):
        res = torch.unique(self.get_cells_for_each_pt(pts), dim=0)
        if list:
            return res.cpu().numpy().tolist()
        else:
            return res

    def get_neighboring_cells(self, cell_idx):
        # todo: use torch.clamp rather than inequalities
        cell_shift = torch.cartesian_prod(torch.arange(0, 3),
                                          torch.arange(0, 3),
                                          torch.arange(0, 3)).to(self.device) - 1
        res = cell_idx + cell_shift
        res[res < 0] = 0
        res[..., 0][res[..., 0] >= self.grid_l] = self.grid_l - 1
        res[..., 1][res[..., 1] >= self.grid_w] = self.grid_w - 1
        res[..., 2][res[..., 2] >= self.grid_h] = self.grid_h - 1

        return torch.unique(res, dim=0)

    def fill_cells(self, pts, features=None, n_point_min=0):
        pts_inside, inside_mask = self.get_pts_in_bounding_box(pts, return_mask=True)
        if features is not None:
            fts_inside = features[inside_mask]
        else:
            fts_inside = None
        cell_indices = self.get_englobing_cells(pts_inside, list=True)
        for cell_idx in cell_indices:
            key = str(cell_idx)
            cell = self.cells[key]
            cell.fill(pts_inside, features=fts_inside, n_point_min=n_point_min)

    def empty_cells(self):
        for key in self.cells:
            cell = self.cells[key]
            cell.empty()

    def get_pt_cloud_from_cells(self, cell_indices, return_features=True):
        do_return_features = return_features and (self.feature_dim > 0)
        pts = torch.zeros(0, 3, device=self.device)
        if do_return_features:
            features = torch.zeros(0, self.feature_dim, device=self.device)

        if len(cell_indices.shape) > 1:
            for cell_index in cell_indices.cpu().numpy().tolist():
                key = str(cell_index)
                cell = self.cells[key]
                pts = torch.vstack((pts, cell.cell_pts))
                if do_return_features:
                    features = torch.vstack((features, cell.cell_features))
        else:
            key = str(cell_indices.cpu().numpy().tolist())
            cell = self.cells[key]
            pts = torch.vstack((pts, cell.cell_pts))
            if do_return_features:
                features = torch.vstack((features, cell.cell_features))

        if do_return_features:
            return pts, features
        else:
            return pts

    def return_entire_pt_cloud(self, return_features=True):
        do_return_features = return_features and (self.feature_dim > 0)

        pts = torch.zeros(0, 3, device=self.device)
        fts = torch.zeros(0, self.feature_dim, device=self.device)

        for key in self.cells:
            cell = self.cells[key]
            pts = torch.vstack((pts, cell.cell_pts))
            if do_return_features:
                fts = torch.vstack((fts, cell.cell_features))

        if do_return_features:
            return pts, fts
        else:
            return pts

    def sample_in_box(self, n_sample):
        return self.x_min + (self.x_max - self.x_min) * torch.rand(n_sample, 3, device=self.device)

    def initialize_proxy_points(self, n_proxy_points=None, default_proba_value=0.5):
        if n_proxy_points is None:
            n_proxy_points = self.n_proxy_points
        self.proxy_points = self.sample_in_box(n_proxy_points)
        self.proxy_proba = torch.zeros(n_proxy_points, 1, device=self.device) + default_proba_value
        self.proxy_supervision_occ = torch.ones(n_proxy_points, 1, device=self.device)
        self.view_states = torch.zeros(n_proxy_points, self.n_view_state_cameras, device=self.device)
        self.out_of_field = torch.ones(n_proxy_points, 1, device=self.device)

        self.proxy_n_inside_fov = torch.zeros(n_proxy_points, 1, device=self.device)
        self.proxy_n_behind_depth = torch.zeros(n_proxy_points, 1, device=self.device)

    def get_proxy_indices_from_mask(self, proxy_mask):
        # all_indices = torch.linspace(start=0,
        #                              end=self.n_proxy_points - 1,
        #                              steps=self.n_proxy_points,
        #                              device=self.device).long().view(-1, 1)
        all_indices = torch.arange(start=0,
                                   end=self.n_proxy_points,
                                   device=self.device).view(-1, 1)

        return all_indices[proxy_mask]

    def get_proxy_mask_from_indices(self, proxy_indices):
        mask = torch.zeros(self.n_proxy_points, device=self.device).bool()
        mask[proxy_indices.view(-1).long()] = True

        return mask

    def update_proxy_view_states(self, camera, proxy_mask,
                                 signed_distances=None, distance_to_surface=None, X_cam=None):
        """
        Update view_state vector for proxy point corresponding to proxy_mask.
        View_state are computed in first camera view space (i.e. fov_camera_0).
        If signed_distances are provided, only the view states of points with
        signed_distances < distance_to_surface will be updated.

        :param camera: (Camera) Current camera used in the scene.
        :param proxy_mask: (Tensor) Mask tensor with shape (n_proxy_points)
        :param signed_distances: (Tensor) Signed distances tensor with shape (1, n_proxy_points_in_mask, 1)
            or (n_proxy_points_in_mask, 1) where n_proxy_point_in_mask = proxy_mask.sum().
        :param distance_to_surface: (float)
        :param X_cam: (Tensor)
        :return: None
        """
        fov_proxy_points = self.proxy_points[proxy_mask]
        # If the depth map is provided, we don't update the view_state of non-visible proxy points,
        # i.e. proxy points behind the depth map.
        # if (depth_map is not None) and (depth_mask is not None):
        #     sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
        #                                                          depth_maps=depth_map,
        #                                                          mask=depth_mask).view(-1)
        if signed_distances is not None:
            sgn_dists = signed_distances.view(-1)
            # if sgn_dists.shape[0] > proxy_mask.sum().long():
            #     raise NameError("signed_distances should have shape (n_proxy_points_in_mask, ...) "
            #                     "or (1, n_proxy_points_in_mask, ...)."
            #                     "It should be computed for the current camera only.")

            if distance_to_surface is None:
                distance_to_surface = 3 * self.distance_between_proxy_points

            update_mask = torch.zeros_like(proxy_mask).bool()
            update_mask[proxy_mask] = False + (sgn_dists < distance_to_surface)
            fov_proxy_points = self.proxy_points[update_mask]
        # Else, we update view_state for every proxy point in the field of view.
        else:
            update_mask = proxy_mask

        if X_cam is None:
            X_cam = camera.X_cam

        # fov_0_transform = camera.fov_camera_0.get_world_to_view_transform()
        # self.view_states[update_mask] += compute_view_state(fov_0_transform.transform_points(fov_proxy_points
        #                                                                                      ).view(1, -1, 3),
        #                                                     fov_0_transform.transform_points(X_cam),
        #                                                     self.view_state_n_elev,
        #                                                     self.view_state_n_azim
        #                                                     ).view(-1, self.view_state_n_elev * self.view_state_n_azim)
        self.view_states[update_mask] += compute_view_state(fov_proxy_points.view(1, -1, 3),
                                                            X_cam,
                                                            self.view_state_n_elev,
                                                            self.view_state_n_azim
                                                            ).view(-1, self.view_state_n_elev * self.view_state_n_azim)

        self.view_states[update_mask] = torch.heaviside(self.view_states[update_mask],
                                                        values=torch.zeros(len(fov_proxy_points),
                                                                           self.n_view_state_cameras,
                                                                           device=self.device))

    def update_proxy_out_of_field(self, fov_proxy_mask):
        """
        Sets out_of_field value to 0. for proxy_points corresponding to fov_proxy_mask.

        :param fov_proxy_mask: (Tensor) Mask tensor with shape (n_proxy_points)
        :return: None
        """
        self.out_of_field[fov_proxy_mask] = 0.

    def update_proxy_supervision_occ(self, proxy_mask, signed_distances, tol=0.):
        """
        Update supervision occupancy value for every proxy points in proxy_mask, based on the current signed distances.

        :param proxy_mask: (Tensor) Mask tensor with shape (n_proxy_points)
        :param signed_distances: (Tensor) Signed distances tensor with shape (1, n_proxy_points_in_mask, 1)
            or (n_proxy_points_in_mask, 1) where n_proxy_point_in_mask = proxy_mask.sum().
            It should be computed using the current camera only.
        :return: None
        """
        # Old method
        # self.proxy_supervision_occ[proxy_mask] *= (signed_distances.view(-1, 1) >= -tol).float()

        # New method
        # proxy_n_inside_fov = self.proxy_n_inside_fov[proxy_mask]
        # proxy_n_behind_depth = self.proxy_n_behind_depth[proxy_mask]
        # proxy_n_inside_fov += 1
        # proxy_n_behind_depth += (signed_distances.view(-1, 1) >= -tol).float()
        # self.proxy_supervision_occ[proxy_mask] = (proxy_n_behind_depth / proxy_n_inside_fov
        #                                           >= self.score_threshold).float()

        self.proxy_n_inside_fov[proxy_mask] += 1
        self.proxy_n_behind_depth[proxy_mask] += (signed_distances.view(-1, 1) >= -tol).float()
        self.proxy_supervision_occ[proxy_mask] = ((self.proxy_n_behind_depth[proxy_mask]
                                                   / self.proxy_n_inside_fov[proxy_mask])
                                                  >= self.score_threshold).float()

    def reset_proxy_supervision_occ(self):
        self.proxy_supervision_occ = torch.ones_like(self.proxy_supervision_occ)

    def get_key_from_idx(self, cell_idx):
        key = str(cell_idx.cpu().numpy().tolist())
        return key

    def scale_points_in_cell_neighborhood(self, cell_idx, pts, prediction_box_diag=1., neighborhood_factor=3.):
        # TO CHANGE !!!!!!!!!!!!!!!!
        key = self.get_key_from_idx(cell_idx)
        cell = self.cells[key]

        center = cell.center
        diag = neighborhood_factor * torch.linalg.norm(cell.x_max - cell.x_min)

        return (pts - center) * prediction_box_diag / diag

    def set_all_features_to_value(self, value):
        """
        Set all features to the provided value.

        :param value: (float or Tensor)
        :return: None
        """
        for key in self.cells:
            cell = self.cells[key]
            if len(cell.cell_features) > 0:
                cell.cell_features = torch.zeros_like(cell.cell_features) + value

    def path_collides(self, params, camera, X_cam):
        '''
        Returns true if this path will lead to collision.

        :param params: (Params)
        :param camera: (Camera)
        :param X_cam: (Tensor) Has shape (1, 3).
        :param oof_collides: (bool) If True, every out-of-field camera poses is supposed to provoke collision.
        '''
        in_bbox = (X_cam >= self.x_min) * (X_cam <= self.x_max)
        in_bbox = in_bbox.prod(dim=-1).bool().item()

        if not in_bbox:
            return False
        
        # Compute the direction vector of the line
        direction = X_cam - camera.X_cam

        # Compute the vector from each point in the point cloud to the line start point
        X, _ = self.return_entire_pt_cloud(return_features=True)
        point_to_start = X - camera.X_cam

        # Compute the perpendicular distance from points to the line (using the point-to-line distance formula)
        distances = torch.norm(torch.cross(direction, point_to_start), dim=1) / torch.norm(direction)

        # Check if there exists any point with a distance less than a threshold
        threshold = 0.1  # Set the threshold value
        intersects = torch.any(distances < threshold)

        return intersects


    def camera_collides(self, params, camera, X_cam, oof_collides=False,
                        collision_n_threshold=6):
        """
        Returns True if camera position X_cam provokes collision.

        :param params: (Params)
        :param camera: (Camera)
        :param X_cam: (Tensor) Has shape (1, 3).
        :param oof_collides: (bool) If True, every out-of-field camera poses is supposed to provoke collision.
        :param collision_n_threshold: (int) Number of close occupied proxy points in the volume above which
            the camera pose is considered occupied. Equal to 12 by default, which is the number of neighbors
            inside a dense 3D sphere packing.
        :return:(bool)
        """

        in_bbox = (X_cam >= self.x_min) * (X_cam <= self.x_max)
        in_bbox = in_bbox.prod(dim=-1).bool().item()

        if not in_bbox:
            return False

        else:
            device = X_cam.get_device()
            ray = camera.X_cam + torch.linspace(0, 1, params.n_interpolation_steps,
                                                device=device).view(-1, 1).expand(-1, 3) * (X_cam - camera.X_cam)

            distances = torch.cdist(self.proxy_points.double(), ray.double()).min(dim=-1)[0]
            dist_mask = distances < self.distance_between_proxy_points

            # distances = torch.linalg.norm(self.proxy_points - X_cam, dim=1)
            # dist_mask = distances[distances < self.distance_between_proxy_points]

            carved_mask = self.proxy_supervision_occ[..., 0] > 0.
            oof_mask = self.out_of_field[..., 0] > 0.

            if oof_collides:
                collision_mask = (oof_mask + carved_mask) * dist_mask
            else:
                collision_mask = (carved_mask * ~oof_mask) * dist_mask

            return collision_mask.sum() > collision_n_threshold

    def camera_coverage_gain(self, part_pc, surface_epsilon=None, surface_epsilon_factor=None):
        """
        Return the number of new points in the surface point cloud covered by the partial point cloud part_pc.
        This function should be used with the surface_scene only, and cell features should represent coverage state
        of points.
        No need to weight points with distance, since the part_pc already reflects
        the loss in sampling density with the distance.

        :param part_pc: (Tensor) Partial point cloud tensor with shape (n, 3).
        :param surface_epsilon: (float) Epsilon value for surface coverage computation.
        :return: (float)
        """
        if surface_epsilon is None:
            epsilon = self.cell_resolution  # or maybe 2 * self.cell_resolution?
        else:
            epsilon = surface_epsilon

        if surface_epsilon_factor is not None:
            epsilon = epsilon * surface_epsilon_factor

        coverage_gain = 0.
        # surface_pts_in_fov = camera.get_points_in_fov(pts=surface_point_cloud,
        #                                               return_mask=False,
        #                                               fov_camera=fov_camera,
        #                                               fov_range=fov_range)
        part_pc_in_box = self.get_pts_in_bounding_box(pts=part_pc,
                                                      return_mask=False)

        cell_indices = self.get_englobing_cells(part_pc_in_box, list=True)
        for cell_idx in cell_indices:
            key = str(cell_idx)
            cell = self.cells[key]
            surface_pts_in_cell = cell.cell_pts
            surface_pts_covered = cell.cell_features.view(-1)
            if (len(surface_pts_in_cell) > 0) and (len(part_pc_in_box) > 0):
                cell_coverage = torch.min(torch.cdist(surface_pts_in_cell.double(),
                                                      part_pc_in_box.double(), p=2.0), dim=-1)[0].float()
                cell_coverage = torch.heaviside(epsilon - cell_coverage,
                                                values=torch.zeros_like(cell_coverage, device=self.device))
                cell_coverage_gain = cell_coverage * (1. - surface_pts_covered)
                coverage_gain += cell_coverage_gain.sum()

        return coverage_gain

    def scene_coverage(self, recovered_scene, surface_epsilon=None):
        
        surface_epsilon = surface_epsilon
        if surface_epsilon == None:
            epsilon = 2. * self.cell_resolution  # or maybe 2 * self.cell_resolution?
        else:
            epsilon = surface_epsilon
        coverage = 0.
        n_gt_pts = 0

        for key in self.cells:
            gt_cell = self.cells[key]
            gt_pts = gt_cell.cell_pts
            if len(gt_pts) > 0:
                n_gt_pts += len(gt_pts)

                rec_cell = recovered_scene.cells[key]
                rec_pts = rec_cell.cell_pts
                if len(rec_pts) > 0:
                    distances = torch.min(torch.cdist(gt_pts.double(),
                                                          rec_pts.double(), p=2.0), dim=-1)[0]
                    
                    coverage_cell = torch.heaviside(epsilon - distances,
                                                    values=torch.zeros_like(distances, device=self.device))

                    coverage += torch.sum(coverage_cell)

        return coverage / n_gt_pts, n_gt_pts

    def get_covered_points(self, recovered_scene, surface_epsilon=None):
        if surface_epsilon == None:
            epsilon = 2. * self.cell_resolution
        else:
            epsilon = surface_epsilon

        coverage = 0.
        n_gt_pts = 0

        covered_pts = torch.zeros(0, 3, device=self.device)
        uncovered_pts = torch.zeros(0, 3, device=self.device)

        for key in self.cells:
            gt_cell = self.cells[key]
            gt_pts = gt_cell.cell_pts
            if len(gt_pts) > 0:
                n_gt_pts += len(gt_pts)

                rec_cell = recovered_scene.cells[key]
                rec_pts = rec_cell.cell_pts
                if len(rec_pts) > 0:
                    coverage_cell = torch.min(torch.cdist(gt_pts, rec_pts, p=2.0), dim=-1)[0]
                    coverage_cell = torch.heaviside(epsilon - coverage_cell,
                                                    values=torch.zeros_like(coverage_cell, device=self.device))
                    coverage += torch.sum(coverage_cell)
                    covered_pts = torch.vstack((covered_pts, gt_pts[coverage_cell > 0.]))
                    uncovered_pts = torch.vstack((uncovered_pts, gt_pts[coverage_cell <= 0.]))
                else:
                    uncovered_pts = torch.vstack((uncovered_pts, gt_pts))

        return coverage / n_gt_pts, n_gt_pts, covered_pts, uncovered_pts


class Memory:
    def __init__(self, scene_memory_paths, n_trajectories, current_epoch, verbose=True):
        self.scene_memory_paths = scene_memory_paths
        self.n_trajectories = n_trajectories
        self.current_epoch = current_epoch

        for scene_memory_path in self.scene_memory_paths:
            # Creating scene memory folder
            try:
                os.mkdir(scene_memory_path)
            except OSError as error:
                if verbose:
                    print(scene_memory_path, "already exists. The model will write new frames over old frames.")

            # Creating training subfolder
            training_path = os.path.join(scene_memory_path, 'training')
            try:
                os.mkdir(training_path)
            except OSError as error:
                if verbose:
                    print(training_path, "already exists. The model will write new frames over old frames.")

            # Creating training trajectories subsubfolder
            for i_traj in range(n_trajectories):
                traj_i_path = os.path.join(training_path, str(i_traj))
                try:
                    os.mkdir(traj_i_path)
                except OSError as error:
                    if verbose:
                        print(traj_i_path, "already exists. The model will write new frames over old frames.")

                # Frames subsubsubfolder
                traj_i_frames_path = os.path.join(traj_i_path, 'frames')
                try:
                    os.mkdir(traj_i_frames_path)
                except OSError as error:
                    if verbose:
                        print(traj_i_frames_path, "already exists. The model will write new frames over old frames.")

                # Occupancy field subsubfolder
                traj_i_occupancy_path = os.path.join(traj_i_path, 'occupancy')
                try:
                    os.mkdir(traj_i_occupancy_path)
                except OSError as error:
                    if verbose:
                        print(traj_i_occupancy_path, "already exists. The model will write new files over old files.")

                # Surface subsubsubfolder
                traj_i_surface_path = os.path.join(traj_i_path, 'surface')
                try:
                    os.mkdir(traj_i_surface_path)
                except OSError as error:
                    if verbose:
                        print(traj_i_surface_path, "already exists. The model will write new files over old files.")

                # Depths subsubsubfolder
                traj_i_depths_path = os.path.join(traj_i_path, 'depths')
                try:
                    os.mkdir(traj_i_depths_path)
                except OSError as error:
                    if verbose:
                        print(traj_i_depths_path, "already exists. The model will write new frames over old frames.")

            training_poses_path = os.path.join(training_path, 'poses')
            try:
                os.mkdir(training_poses_path)
            except OSError as error:
                if verbose:
                    print(error)

        # The below code is for avoiding the creation of the files when testing
        # for scene_memory_path in self.scene_memory_paths:
        #     # Creating scene memory folder
        #     try:
        #         os.mkdir(scene_memory_path)
        #     except OSError as error:
        #         print(scene_memory_path, "already exists. Clearing contents.")
        #         shutil.rmtree(scene_memory_path)  # Remove existing directory
        #         os.mkdir(scene_memory_path)  # Recreate it as empty

        #     # Creating training subfolder
        #     training_path = os.path.join(scene_memory_path, 'training')
        #     os.mkdir(training_path)  # Directly create, assuming parent was just cleaned

        #     # Creating training trajectories subsubfolder
        #     for i_traj in range(n_trajectories):
        #         traj_i_path = os.path.join(training_path, str(i_traj))
        #         os.mkdir(traj_i_path)  # Directly create, clean start

        #         # Frames subsubsubfolder
        #         traj_i_frames_path = os.path.join(traj_i_path, 'frames')
        #         os.mkdir(traj_i_frames_path)  # Directly create, clean start

        #         # Occupancy field subsubfolder
        #         traj_i_occupancy_path = os.path.join(traj_i_path, 'occupancy')
        #         os.mkdir(traj_i_occupancy_path)  # Directly create, clean start

        #         # Surface subsubsubfolder
        #         traj_i_surface_path = os.path.join(traj_i_path, 'surface')
        #         os.mkdir(traj_i_surface_path)  # Directly create, clean start

        #         # Depths subsubsubfolder
        #         traj_i_depths_path = os.path.join(traj_i_path, 'depths')
        #         os.mkdir(traj_i_depths_path)  # Directly create, clean start

        #     training_poses_path = os.path.join(training_path, 'poses')
        #     os.mkdir(training_poses_path)  # Directly create, assuming parent was just cleaned

    def get_memory_size(self):
        """
        Return the number of frames in the whole memory.
        :return: (int)
        """
        memory_size = 0
        for scene_memory_path in self.scene_memory_paths:
            for i in range(self.n_trajectories):
                frames_dir_path = self.get_trajectory_frames_path(scene_memory_path, i)
                memory_size += len(os.listdir(frames_dir_path))
        return memory_size

    def get_trajectory_frames_path(self, scene_memory_path, trajectory_nb):
        """
        Return the path to the directory containing the frame files for a given trajectory of a scene.

        :param scene_memory_path: (string) Path to the memory folder of a scene.
        :param trajectory_nb: (int) Number of the trajectory.
        :return: (string) Path to the directory containing the frame files.
        """
        if scene_memory_path not in self.scene_memory_paths:

            raise NameError("The path provided is not registered in memory.")
        return_path = os.path.join(scene_memory_path, 'training')
        return_path = os.path.join(return_path, str(trajectory_nb))
        return_path = os.path.join(return_path, 'frames')

        return return_path

    def get_trajectory_occupancy_path(self, scene_memory_path, trajectory_nb):
        """
        Return the path to the directory containing the occupancy field files for a given trajectory of a scene.

        :param scene_memory_path: (string) Path to the memory folder of a scene.
        :param trajectory_nb: (int) Number of the trajectory.
        :return: (string) Path to the directory containing the occupancy field files.
        """
        if scene_memory_path not in self.scene_memory_paths:
            raise NameError("The path provided is not registered in memory.")

        return_path = os.path.join(scene_memory_path, 'training')
        return_path = os.path.join(return_path, str(trajectory_nb))
        return_path = os.path.join(return_path, 'occupancy')

        return return_path

    def get_trajectory_surface_path(self, scene_memory_path, trajectory_nb):
        """
        Return the path to the directory containing the surface files for a given trajectory of a scene.

        :param scene_memory_path: (string) Path to the memory folder of a scene.
        :param trajectory_nb: (int) Number of the trajectory.
        :return: (string) Path to the directory containing the surface files.
        """
        if scene_memory_path not in self.scene_memory_paths:
            raise NameError("The path provided is not registered in memory.")

        return_path = os.path.join(scene_memory_path, 'training')
        return_path = os.path.join(return_path, str(trajectory_nb))
        return_path = os.path.join(return_path, 'surface')

        return return_path

    def get_trajectory_depths_path(self, scene_memory_path, trajectory_nb):
        """
        Return the path to the directory containing the predicted depth files for a given trajectory of a scene.

        :param scene_memory_path: (string) Path to the memory folder of a scene.
        :param trajectory_nb: (int) Number of the trajectory.
        :return: (string) Path to the directory containing the predicted depth files.
        """
        if scene_memory_path not in self.scene_memory_paths:
            raise NameError("The path provided is not registered in memory.")

        return_path = os.path.join(scene_memory_path, 'training')
        return_path = os.path.join(return_path, str(trajectory_nb))
        return_path = os.path.join(return_path, 'depths')

        return return_path

    def get_poses_path(self, scene_memory_path):
        return_path = os.path.join(scene_memory_path, 'training')
        return_path = os.path.join(return_path, 'poses')

        return return_path

    def get_random_batch_for_depth_model(self, params, camera, n_sample, alphas, mode='supervision'):
        min_alpha = -params.n_alpha
        max_alpha = alphas[-1]
        if max_alpha < 0:
            max_alpha = 0
        n_alpha = len(alphas)

        replace = n_sample <= len(self.scene_memory_paths)
        replace = True
        sample_memory_paths = np.random.choice(self.scene_memory_paths, size=n_sample, replace=replace)

        batch_images = []
        batch_mask = []
        batch_R = []
        batch_T = []
        batch_zfar = []

        alpha_images = []
        alpha_mask = []
        alpha_R = []
        alpha_T = []
        alpha_zfar = []

        for memory_path in sample_memory_paths:
            # n_traj is the number of trajectories currently in the folder
            n_traj = min(self.current_epoch+1, self.n_trajectories)
            # We want to select any trajectory except the current one
            i_traj = (self.current_epoch + np.random.randint(low=1, high=n_traj)) % n_traj
            # i_traj = np.random.randint(low=0, high=n_traj) % n_traj

            frames_path = self.get_trajectory_frames_path(memory_path, i_traj)

            min_frame_nb = max_alpha - min_alpha
            if i_traj == self.current_epoch % self.n_trajectories:
                max_frame_nb = camera.n_frames_captured - params.n_interpolation_steps
                raise NameError("APOCALYPSE!")
            else:
                max_frame_nb = params.n_interpolation_steps * \
                               (params.n_poses_in_trajectory + 1) + 1  # len(os.listdir(frames_path))

            frame_nb = np.random.randint(low=min_frame_nb, high=max_frame_nb)

            s_images, s_mask, s_R, s_T, s_zfar = load_images_for_depth_model(camera,
                                                                             n_frames=1,
                                                                             n_alpha=n_alpha,
                                                                             frame_nb=frame_nb,
                                                                             frames_dir_path=frames_path)

            batch_dict, alpha_dict = create_batch_for_depth_model(params, s_images, s_mask, s_R, s_T, s_zfar,
                                                                  mode, camera.device)

            batch_images.append(batch_dict['images'])
            batch_mask.append(batch_dict['mask'])
            batch_R.append(batch_dict['R'])
            batch_T.append(batch_dict['T'])
            batch_zfar.append(batch_dict['zfar'])

            alpha_images.append(alpha_dict['images'])
            alpha_mask.append(alpha_dict['mask'])
            alpha_R.append(alpha_dict['R'])
            alpha_T.append(alpha_dict['T'])
            alpha_zfar.append(alpha_dict['zfar'])

        batch_dict = {'images': torch.cat(batch_images, dim=0),
                      'mask': torch.cat(batch_mask, dim=0).bool(),
                      'R': torch.cat(batch_R, dim=0),
                      'T': torch.cat(batch_T, dim=0),
                      'zfar': torch.cat(batch_zfar, dim=0)}

        alpha_dict = {'images': torch.cat(alpha_images, dim=0),
                      'mask': torch.cat(alpha_mask, dim=0).bool(),
                      'R': torch.cat(alpha_R, dim=0),
                      'T': torch.cat(alpha_T, dim=0),
                      'zfar': torch.cat(alpha_zfar, dim=0)}

        return batch_dict, alpha_dict

    def get_random_scene_for_scone_model(self, params, camera, device, n_weights_update,
                                         sample_memory_path=None):
        """
        Create and return all objects needed to reproduce training iterations for occupancy and coverage gain modules
        in a 3D scene.
        The scene is selected randomly in the memory, unless a specific sample_memory_path is provided.

        :param params: (Params)
        :param camera: (Camera)
        :param device: (Device)
        :param n_weights_update: (int)
        :param sample_memory_path: (string) None by default.
        :return:
        """
        # We select a scene
        if sample_memory_path is None:
            # sample_memory_path = np.random.choice(self.scene_memory_paths, size=1)[0]
            sample_memory_path = self.scene_memory_paths[np.random.randint(len(self.scene_memory_paths))]

        # n_traj is the number of trajectories currently in the folder
        n_traj = min(self.current_epoch+1, self.n_trajectories)
        # We want to select any trajectory except the current one
        i_traj = (self.current_epoch + np.random.randint(low=1, high=n_traj)) % n_traj

        # Get data paths
        occupancy_dir_path = self.get_trajectory_occupancy_path(sample_memory_path, i_traj)
        surface_dir_path = self.get_trajectory_surface_path(sample_memory_path, i_traj)
        depths_dir_path = self.get_trajectory_depths_path(sample_memory_path, i_traj)

        # ----------Load supervision data from memory-------------------------------------------------------------------
        # Load surface scene
        total_surface_scene = load_surface_scene_from_memory(surface_dir_path, device)

        # Load proxy scene
        proxy_scene = load_occupancy_field_from_memory(occupancy_dir_path, device)

        # Extract the pseudo-GT occupancy values
        pseudo_gt_proxy_proba = 0. + proxy_scene.proxy_supervision_occ
        proxy_scene.proxy_supervision_occ = torch.ones(proxy_scene.n_proxy_points, 1, device=device)

        # ----------Recreate a valid input for prediction---------------------------------------------------------------
        # -----1. Select a sequence of depth maps
        # Total number of depths in memory directory
        total_depth_nb = len(os.listdir(depths_dir_path))

        # Max number of depths used to recreate a partial pc
        max_depth_nb = params.n_max_memory_depths_for_partial_pc

        # Number of depths in the trajectory used to recreate the partial pc
        traj_depth_nb = np.random.randint(1, max_depth_nb)

        # Number of depths involved in the memory replay process
        involved_depth_nb = traj_depth_nb + n_weights_update * params.n_poses_in_memory_scene_loops
        if involved_depth_nb > total_depth_nb:
            raise NameError("The quantity traj_depth_nb + n_weights_update * params.n_poses_in_memory_scene_loops "
                            "is too big regarding the number of depth maps stored into the memory.")

        # Index for the starting depth map
        start_depth_i = np.random.randint(total_depth_nb - involved_depth_nb + 1)
        depth_list = []

        # -----2. Generate a partial surface point cloud
        # Creating a partial surface scene
        partial_surface_scene = create_scene_like(total_surface_scene)
        full_pc = torch.zeros(0, 3, device=device)
        prediction_camera = None  # This fov_camera object will be used for prediction

        for i in range(traj_depth_nb):
            depth_i = start_depth_i + i
            depth_list.append(depth_i)
            depth_dict = torch.load(os.path.join(depths_dir_path, str(depth_i) + '.pt'), map_location=device)
            depth = depth_dict['depth']
            mask = depth_dict['mask']
            error_mask = depth_dict['error_mask']

            fov_camera = camera.get_fov_camera_from_RT(R_cam=depth_dict['R'][0:0 + 1],
                                                       T_cam=depth_dict['T'][0:0 + 1])
            prediction_camera = fov_camera
            # TO CHANGE: filter points based on SSIM value!
            part_pc = camera.compute_partial_point_cloud(depth=depth[0:0 + 1],
                                                         mask=(mask * error_mask)[0:0 + 1],
                                                         fov_cameras=fov_camera,
                                                         gathering_factor=params.gathering_factor,
                                                         fov_range=params.sensor_range)

            # Fill partial surface scene
            # part_pc_features = torch.ones(len(part_pc), 1, device=device)
            # partial_surface_scene.fill_cells(part_pc, features=part_pc_features)
            full_pc = torch.vstack((full_pc, part_pc))

            # Get Proxy Points in FoV
            fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points,
                                                                        return_mask=True,
                                                                        fov_camera=fov_camera,
                                                                        fov_range=params.sensor_range)

            # Fill cells with proxy points
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
            proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))

            # Computing signed distance of proxy points in fov
            sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points,
                                                                 depth_maps=depth,
                                                                 mask=mask,
                                                                 fov_camera=fov_camera)

            # Updating view_state vectors
            proxy_scene.update_proxy_view_states(camera, fov_proxy_mask,
                                                 signed_distances=sgn_dists,
                                                 distance_to_surface=None, X_cam=fov_camera.get_camera_center())

            # Update supervision occupancy probabilities
            proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)

            # Update the out-of-field status for proxy points inside camera field of view
            proxy_scene.update_proxy_out_of_field(fov_proxy_mask)

        fill_surface_scene(partial_surface_scene, full_pc,
                           random_sampling_max_size=params.n_gt_surface_points,
                           min_n_points_per_cell_fill=3,
                           progressive_fill=params.progressive_fill,
                           max_n_points_per_fill=params.max_points_per_progressive_fill)

        result = {'depths_dir_path': depths_dir_path,
                  'partial_surface_scene': partial_surface_scene,
                  'total_surface_scene': total_surface_scene,
                  'proxy_scene': proxy_scene,
                  'pseudo_gt_proxy_proba': pseudo_gt_proxy_proba,
                  'prediction_camera': prediction_camera,
                  'traj_depth_nb': traj_depth_nb,
                  'start_depth_i': start_depth_i,
                  'depth_list': depth_list}

        return result
