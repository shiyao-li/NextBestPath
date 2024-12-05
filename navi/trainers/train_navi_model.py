import sys
import json
import time
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from macarons.utility.macarons_utils import setup_device, get_dataloader
from torch.optim.lr_scheduler import StepLR
from navi.networks.navi_model import *
from navi.networks.navi_rl_model import *
from ..utility.navi_utils import *


import lmdb
import msgpack
import msgpack_numpy as m

macarons_weights_path = "./weights/macarons/trained_macarons.pth"
navi_weights_path = "./weights/navi"
navi_training_posses = "./training_log"
navi_training_name = "shiyao_doom_4.json"
navi_weights = "./weights/navi/shiyao.pth"

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


def run_training_navi(params=None):
    device = setup_device(params=params)

    dataset_path = params.data_path

    print("data_path: ", dataset_path)
    # Create dataloader from macarons
    world_size, rank = None, None
    scene_name = get_subfolder_names(dataset_path)

    train_dataloader, _, _ = get_dataloader(train_scenes=scene_name,
                                            val_scenes=params.val_scenes,
                                            test_scenes=params.test_scenes,
                                            batch_size=1,
                                            ddp=params.ddp, jz=params.jz,
                                            world_size=world_size, ddp_rank=rank,
                                            data_path=dataset_path)
    
    
    # Create model
    # navi = load_pretrained_navi(pretrained_model_path=params.pretrained_model_path, 
    #                             load_pretrain=params.load_pretrain, 
    #                             device=device)
    


    print("Current learning rate: ", params.navi_lr)
    
    # navi_memory_buffer = ReplayBuffer()

    db_name = 'doom_4.lmdb'
    db_path = os.path.join( "./navi/db", db_name)
    db_env = lmdb.open(db_path, map_size=200 * 1024**3)

    
    # Current path: Macarons-main
    print("Model name: ", params.navi_model_name)
    # print("Numbers of parameters: ", count_parameters(navi))
    # print("Training data: ", len(train_dataloader))
    print("Batch size: ", params.batch_size)
    print("Name of optimizer: ", params.opt_name)
    
    macarons = setup_macarons(params, macarons_weights_path, device)
    memory = setup_memory(params, scene_name, train_dataloader)


    # macarons, memory = setup_memory_and_macarons(params, macarons_weights_path, train_dataloader, device)

    results_to_save = {}
    training_process_path = os.path.join(navi_training_posses, navi_training_name)
    
    grid_size = (256, 256)
    grid_range = (-40, 40) # 70, 70
    heatmap_grid_size = (64, 64)
    kernel_size = 3
    average_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

    center_weight = 2.0 
    total_weight = average_kernel.sum() + center_weight - 1  
    average_kernel[0, 0, kernel_size//2, kernel_size//2] = center_weight 

    average_kernel /= total_weight
    average_kernel = average_kernel.to(device)

    navi = UNet().to(device)
    navi, optimizer, best_loss, start_epoch = initialize_navi(params, navi,
                                            torch_seed=params.torch_seed,
                                            initialize=params.start_from_scratch,
                                            pretrained=True,
                                            ddp_rank=rank)
    
    # chechp = torch.load(navi_weights)
    # navi.load_state_dict(chechp['model_state_dict'])
    # navi.eval()
    
    # navi_ONE = UNet().to(device)
    # navi_ONE, optimizer, best_loss, start_epoch = initialize_navi(params, navi_ONE,
    #                                         torch_seed=params.torch_seed,
    #                                         initialize=params.start_from_scratch,
    #                                         pretrained=True,
    #                                         ddp_rank=rank)
    
    # navi_ONE.train()
    
    best_validation_loss = 1000
    m.patch()
    for current_epoch in range(0, 100):
        
        t = current_epoch
        print("\n-------------------------------------------------------------------------------")
        print(f"Epoch {t}")
        print("-------------------------------------------------------------------------------\n")
        coverage_after_trajectory = []
        
        # base_img_path = "/home/sli/MACARONS-main/navi/train_images_partial_value_model/"
        # folder_img_name = str(current_epoch)
        # folder_img_path = os.path.join(base_img_path, folder_img_name) + "/"
        # os.makedirs(folder_img_path, exist_ok=True)

        folder_img_path = None
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


        torch.cuda.empty_cache()
        # wori!
        navi.eval()
        print("Collection!")
        trajectory_collection(params, current_epoch, train_dataloader, db_env, grid_size, heatmap_grid_size, grid_range, kernel_size, average_kernel,
                navi, coverage_after_trajectory, macarons, memory, device, folder_img_path)
                

        if current_epoch == 0:
            validaion_data = store_validation_data(db_env)
            print("Validation data stored!:. ", len(validaion_data))
        else:
            navi.train()
            print("Training!")
            training_loss, average_validation_loss = train_navi(db_env, params, optimizer, navi, device, folder_img_path, current_epoch, validaion_data)
            
                
        # average_coverage = sum(coverage_after_trajectory)/len(coverage_after_trajectory)

        # print("average_coverage: ",average_coverage)
        torch.cuda.empty_cache()

        # if True:
        #     navi = UNetTransformer().to(device)
        #     navi, optimizer, best_loss, start_epoch = initialize_navi(params, navi,
        #                                             torch_seed=params.torch_seed,
        #                                             initialize=params.start_from_scratch,
        #                                             pretrained=True,
        #                                             ddp_rank=rank)
        # if current_epoch == 2:  
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.00004
        # navi.train()

        # training_loss, last_experience_num = train_navi(navi_memory_buffer, params, optimizer, navi, device, folder_img_path, current_epoch)
        
        # validation process for potential model
        # if True:
        #     coverage_after_trajectory = []
        #     navi.eval()
        #     for tra_num in range(3):
        #         for root, dirs, files in os.walk('./data/2d_rl_data/2d_room_33/macarons_memory', topdown=False):
        #             for name in files:
        #                 file_path = os.path.join(root, name)
        #                 os.remove(file_path)
        #         trajectory_validation(params, current_epoch, train_dataloader, grid_size, grid_range, kernel_size, average_kernel,
        #                 navi, coverage_after_trajectory, macarons, memory, device, tra_num, folder_img_path)
            
        #     average_coverage_validation = sum(coverage_after_trajectory)/len(coverage_after_trajectory)
        #     print("average_coverage_ONE: ",average_coverage_validation)
        
        # Clean old experience
        # t1 = time.time()
        # print("Training done for epoch", t, ".")
        # print(t1-t0)


        if current_epoch > 0:
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                model_filename = "doom4_best_validation_loss.pth".format(params.navi_model_name, training_loss)
                model_save_path = os.path.join(navi_weights_path, model_filename)
                print(model_save_path)
                torch.save({
                    'epoch': t ,
                    'model_state_dict': navi.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'val_losses': val_losses,
                }, model_save_path)

            # last_training_loss = training_loss

            if t % 3 == 0:
                print("Save model")
                model_filename = "doom4_{}_Epoch{:.0f}.pth".format(params.navi_model_name, t)
                model_save_path = os.path.join(navi_weights_path, model_filename)
                print(model_save_path)
                torch.save({
                    'epoch': t ,
                    'model_state_dict': navi.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'val_losses': val_losses,
                }, model_save_path)




        #     # validation_loss = 0
        #     # average_coverage_validation = 0
            results_to_save[t] = {}
        #     # results_to_save[t]["training_data"] = len(navi_memory_buffer.buffer)
            results_to_save[t]["training_loss"] = training_loss
            results_to_save[t]["validation_loss"] = average_validation_loss
        # # # results_to_save[t]["validation_average_coverage"] = average_coverage_validation
        # # # results_to_save[t]["validation_loss"] = validation_loss
        # # # results_to_save[t]["training_time"] = t1-t0 
            with open(training_process_path, 'w') as outfile:
                json.dump(results_to_save, outfile)
        
        # if (sum(training_losses) / len(training_losses)) < best_loss:
        #     best_model_save_path = "best_un_" + params.navi_model_name + ".pth"
        #     best_model_save_path = os.path.join(navi_weights_path, best_model_save_path)
        #     best_loss = sum(training_losses) / len(training_losses)
        #     torch.save({
        #     'epoch': t + 1,
        #     'model_state_dict': navi.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': training_losses
        #     # 'val_losses': val_losses,
        #     }, best_model_save_path)
        #     print("Best loss model saved...", best_loss)
