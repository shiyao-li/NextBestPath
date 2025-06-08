<div align="center">
<h1>NextBestPath: Efficient 3D Mapping of Unseen Environments
</h1>

<a href="https://arxiv.org/pdf/2502.05378" style="margin-right: 10px;">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
<a href="https://shiyao-li.github.io/nbp/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>



*A method for generating the next-best-path for efficient active mapping, along with a new benchmark tailored for complex indoor environments.*

</div>

```bibtex
@inproceedings{li2025nextbestpath,
      title={NextBestPath: Efficient 3D Mapping of Unseen Environments},
      author={Shiyao Li and Antoine Guedon and Cl{\'e}mentin Boittiaux and Shizhe Chen and Vincent Lepetit},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=7WaRh4gCXp}
}
```

## Updates
- [June, 2025] Release the training and test code

## Quick Start

### Prerequisites

First, ensure you have conda installed, then set up the environment:

```bash
# Clone this repository
git clone https://github.com/shiyao-li/NextBestPath.git
cd NextBestPath

# Create and activate conda environment
conda env create -f environment.yml
conda activate exploration
```

### Installation

1. **Download the AiMDoom Dataset**
   
   Download the complete dataset from [Google Drive](https://drive.google.com/drive/folders/1fwhCrxmrJnpdK-egawoX2OYHUxnxAwr-):
   - AiMDoom dataset (4 difficulty levels)
   - Pre-trained NBP model weights

2. **Set up model weights**
   
   Place the downloaded NBP model weights in the following structure:
   ```
   ./weights/nbp/
   ├── AiMDoom_simple_best_val.pth   
   ├── AiMDoom_normal_best_val.pth   
   ├── AiMDoom_hard_best_val.pth   
   └── AiMDoom_insane_best_val.pth 
   ```
<!-- 
## Usage

### Configuration

Before running the navigation system, modify the configuration file `test_via_navi_model.json` to match your setup:

```json
{
  "dataset_path": "/path/to/aimdoom/dataset",
  "model_weights": "./weights/navi/doom1_weights.pth",
  "difficulty_level": "simple",
  "num_camera_poses": 101,
  "use_perfect_depth": true
}
```

### Running Navigation Planning

Execute the main navigation planning script:

```bash
python test_navi_planning_2d.py
```

### Example Usage

```python
import torch
from navigation.navi_planner import NavigationPlanner
from utils.config_loader import load_config

# Load configuration
config = load_config("test_via_navi_model.json")

# Initialize navigation planner
planner = NavigationPlanner(config)

# Load model weights (example for simple difficulty)
planner.load_weights("./weights/navi/doom1_weights.pth")

# Run navigation planning
results = planner.plan_navigation(
    scene_data=your_scene_data,
    num_poses=101
)
``` -->

<!-- 
## Acknowledgments

This work builds upon several excellent projects:

- [MACARONS](https://github.com/Anttwo/MACARONS) - For the foundational depth estimation framework
- AiMDoom Dataset - For providing comprehensive navigation evaluation scenarios -->

