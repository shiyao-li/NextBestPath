import argparse
import os

from macarons.utility.macarons_utils import load_params
from next_best_path.trainers.train_nbp_model import run_training_nbp

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "./data/")
weights_dir = os.path.join(dir_path, "./weights/nbp")
configs_dir = os.path.join(dir_path, "./configs/nbp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train the nbp model in large 3D scenes')
    parser.add_argument('-c', '--config', type=str, help='name of the cofigh file.')
    
    args = parser.parse_args()
    
    if args.config:
        json_name = args.config
    else:
        json_name = "nbp_default_training_config.json"
    
    json_path = os.path.join(configs_dir, json_name)
    params = load_params(json_path)
    
    if params.ddp:
        pass
    elif params.jz:
        pass
    else:
        run_training_nbp(params=params)
    
    