import sys
import json
import time
import os
import torch

from macarons.utility.macarons_utils import setup_device, get_dataloader
from torch.optim.lr_scheduler import StepLR
from next_best_path.networks.nbp_model import *
from ..utility.nbp_utils import *


import lmdb
import msgpack_numpy as m

nbp_weights_path = "./weights/nbp"
nbp_training_posses = "./training_log"
nbp_training_name = "AiMDoom_insane.json"
nbp_weights = "./weights/nbp/....pth"

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


def run_training_nbp(params=None):
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
    


    print("Current learning rate: ", params.nbp_lr)

    db_name = 'AiMDoom_insane.lmdb'
    db_path = os.path.join( "./nbp/db", db_name)
    db_env = lmdb.open(db_path, map_size=200 * 1024**3)

    
    print("Model name: ", params.nbp_model_name)
    print("Batch size: ", params.batch_size)
    print("Name of optimizer: ", params.opt_name)

    memory = setup_memory(params, scene_name, train_dataloader)

    results_to_save = {}
    training_process_path = os.path.join(nbp_training_posses, nbp_training_name)
    
    pc2img_size = (256, 256)
    prediction_range = (-40, 40) # 70, 70
    value_map_size = (64, 64)

    nbp = NBP().to(device)
    nbp, optimizer, best_loss, start_epoch = initialize_nbp(params, nbp,
                                            torch_seed=params.torch_seed,
                                            initialize=params.start_from_scratch,
                                            pretrained=True,
                                            ddp_rank=rank)
    
    # chechp = torch.load(nbp_weights)
    # nbp.load_state_dict(chechp['model_state_dict'])
    # nbp.eval()
    
    best_validation_loss = 1000
    m.patch()
    for current_epoch in range(0, 100):
        
        t = current_epoch
        print("\n-------------------------------------------------------------------------------")
        print(f"Epoch {t}")
        print("-------------------------------------------------------------------------------\n")
        coverage_after_trajectory = []
        

        folder_img_path = None
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


        torch.cuda.empty_cache()
        nbp.eval()
        print("Collection!")
        trajectory_collection(params, current_epoch, train_dataloader, db_env, pc2img_size, value_map_size, prediction_range,
                nbp, coverage_after_trajectory, memory, device, folder_img_path)
                

        if current_epoch == 0:
            validaion_data = store_validation_data(db_env)
            print("Validation data stored!:. ", len(validaion_data))
        else:
            nbp.train()
            print("Training!")
            training_loss, average_validation_loss = train_nbp(db_env, params, optimizer, nbp, device, folder_img_path, current_epoch, validaion_data)
            

        torch.cuda.empty_cache()


        if current_epoch > 0:
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                model_filename = "AiMDoom_insane_best_val.pth".format(params.nbp_model_name, training_loss)
                model_save_path = os.path.join(nbp_weights_path, model_filename)
                print(model_save_path)
                torch.save({
                    'epoch': t ,
                    'model_state_dict': nbp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'val_losses': val_losses,
                }, model_save_path)

            # last_training_loss = training_loss

            if t % 3 == 0:
                print("Save model")
                model_filename = "AiMDoom_insane_{}_Epoch{:.0f}.pth".format(params.nbp_model_name, t)
                model_save_path = os.path.join(nbp_weights_path, model_filename)
                print(model_save_path)
                torch.save({
                    'epoch': t ,
                    'model_state_dict': nbp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'val_losses': val_losses,
                }, model_save_path)


            results_to_save[t] = {}
            results_to_save[t]["training_loss"] = training_loss
            results_to_save[t]["validation_loss"] = average_validation_loss

            with open(training_process_path, 'w') as outfile:
                json.dump(results_to_save, outfile)
        
