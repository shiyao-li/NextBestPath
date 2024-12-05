import argparse
import os
import sys

sys.path.append(os.path.abspath('../'))
from macarons.testers.navi_planning_2d import *

dir_path = os.path.abspath(os.path.dirname(__file__))
test_configs_dir = os.path.join(dir_path, "./configs/test/")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test a full macarons model and show recontructed 3D scenes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "test_in_default_scenes_config.json".')

    args = parser.parse_args()

    if args.config:
        params_name = args.config
    else:
        params_name = "test_via_navi_model.json"

    params_name = os.path.join(test_configs_dir, params_name)
    test_params = load_params(params_name)
    numGPU=test_params.numGPU

    with torch.no_grad():
        test_navi_planning_2d(params_file=test_params.params_name,
                 model_file=test_params.model_name,
                 results_json_file=test_params.results_json_name,
                 numGPU=test_params.numGPU,
                 test_scenes=test_params.test_scenes,
                 test_resolution=test_params.test_resolution,
                 use_perfect_depth_map=test_params.use_perfect_depth_map,
                 compute_collision=test_params.compute_collision,
                 load_json=test_params.load_json,
                 dataset_path=test_params.dataset_path,
                 navi_weights=test_params.navi_weights)
