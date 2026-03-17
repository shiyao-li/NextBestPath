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
    subfolders = []  # 存放子文件夹名字的列表
    # 遍历文件夹下的所有子文件夹以及文件
    for entry in os.listdir(folder_path):
        # 构造子文件夹的完整路径
        full_path = os.path.join(folder_path, entry)
        # 检查该路径是否为文件夹
        if os.path.isdir(full_path):
            subfolders.append(entry)  # 将子文件夹名字添加到列表中
    
    return subfolders


def run_training_nbp(params=None):
    device = setup_device(params=params)
    dataset_path = params.data_path

    print("数据加载路径为: ", dataset_path)
    # Create dataloader from macarons
    world_size, rank = None, None # 分布式训练时，world_size为多少个进程参数分布式训练，rank为当前进程的排名
    scene_name = get_subfolder_names(dataset_path) # 获取数据集中的所有子文件夹名字

    train_dataloader, _, _ = get_dataloader(train_scenes=scene_name, # 训练场景名
                                            val_scenes=params.val_scenes, # 验证场景名
                                            test_scenes=params.test_scenes, # 测试场景名
                                            batch_size=1, # 批量大小
                                            ddp=params.ddp, jz=params.jz, # 是否开启分布式训练
                                            world_size=world_size, ddp_rank=rank, # 分布式训练时，world_size为多少个进程参数分布式训练，rank为当前进程的排名
                                            data_path=dataset_path) # 数据加载路径
    


    print("当前学习率为: ", params.nbp_lr)

    db_name = 'AiMDoom_insane.lmdb'# 存放经验的数据库名
    db_path = os.path.join( "./nbp/db", db_name) # 存放经验的数据库路径
    db_env = lmdb.open(db_path, map_size=200 * 1024**3) # 打开数据库

    
    print("模型名称为: ", params.nbp_model_name)
    print("批量大小为: ", params.batch_size)
    print("优化器名称为: ", params.opt_name)

    memory = setup_memory(params, scene_name, train_dataloader) # 设置记忆

    results_to_save = {} # 存放训练结果
    training_process_path = os.path.join(nbp_training_posses, nbp_training_name) # ./training_log\AiMDoom_insane.json
    
    pc2img_size = (256, 256) # 点云转图像大小
    prediction_range = (-40, 40) # 预测范围,单位是m
    value_map_size = (64, 64) # 价值地图大小

    nbp = NBP().to(device) # 初始化NBP模型
    nbp, optimizer, best_loss, start_epoch = initialize_nbp(params, nbp,
                                            torch_seed=params.torch_seed,
                                            initialize=params.start_from_scratch,
                                            pretrained=True,
                                            ddp_rank=rank)
    
    # chechp = torch.load(nbp_weights)
    # nbp.load_state_dict(chechp['model_state_dict'])
    # nbp.eval()
    
    best_validation_loss = 1000
    m.patch() # 让msgpack支持numpy,后面用msgpack读写LMDB里面的经验（含大量numpy），防止msgpack无法处理numpy类型
    # 训练都是100个epoch，先收集数据，因为NBP的训练是基于经验回放的，所以需要大量的经验来训练模型
    for current_epoch in range(0, 100):
        
        t = current_epoch
        print("\n-------------------------------------------------------------------------------")
        print(f"当前训练第 {t} 个epoch")
        print("-------------------------------------------------------------------------------\n")
        coverage_after_trajectory = [] #每条轨迹在“规定步数”结束时，场景被覆盖的百分比。
        

        folder_img_path = None # 存放图片的路径
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # 让CUDA的错误信息更详细


        torch.cuda.empty_cache() # 清空CUDA缓存
        nbp.eval() # 设置模型为评估模式，不进行训练
        print("开始收集数据!")
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
        
