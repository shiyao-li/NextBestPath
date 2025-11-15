<div align="center">
<h1>NextBestPath: Efficient 3D Mapping of Unseen Environments</h1>

[Shiyao Li](https://shiyao-li.github.io/), [Antoine Guédon](https://anttwo.github.io/), [Clémentin Boittiaux](https://clementinboittiaux.github.io/), [Shizhe Chen](https://cshizhe.github.io/), [Vincent Lepetit](https://vincentlepetit.github.io/)

<a href="https://arxiv.org/pdf/2502.05378" style="margin-right: 10px;">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper">
</a>
<a href="https://shiyao-li.github.io/nbp/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

*A method for generating the next-best-path for efficient active mapping, along with a new benchmark tailored for complex indoor environments.*

</div>


##### 🌟 If you find our work helpful, please consider giving a ⭐️ to this repository and citing our paper!



## 🗺️ Project Overview

NextBestPath (NBP) is a novel method for next-best-path planning in 3D scene exploration. Unlike previous methods, NBP is designed to directly maximize mapping efficiency and coverage along the camera trajectory.


This repository contains:
* A simulator based on PyTorch3D and Trimesh
* Functions for generating ground truth point clouds from meshes and evaluating reconstructed point clouds
* Scripts for testing and training NBP models on AiMDoom dataset.

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
- Todo: Release the models of MACARONS and the corresponding scripts


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
   - The toolkit and code to build AiMDoom dataset: [Github_link](https://github.com/shiyao-li/AiMDoom)

2. **Download and set up model weights**
   
   Download NBP models from [Google Drive](https://drive.google.com/drive/folders/1jAEKrznbbZ5bwu39y0ah4pszMlTuVAfH?usp=sharing), and put them under the `./weights/nbp/` folder.
   
   Place the downloaded NBP model weights in the following structure:
   ```
   ./weights/nbp/
   ├── AiMDoom_simple_best_val.pth  
   ├── AiMDoom_normal_best_val.pth  
   ├── AiMDoom_hard_best_val.pth  
   └── AiMDoom_insane_best_val.pth
   ```

### Usage

1. **Configs**
   
   All config files are under the `./configs/` folder.

2. **Test NBP method**
   ```bash
   python test_nbp_planning.py
   ```

3. **Train NBP models**
   ```bash
   python train_nbp.py
   ```

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{li2025nextbestpath,
  title={NextBestPath: Efficient 3D Mapping of Unseen Environments},
  author={Shiyao Li and Antoine Guedon and Cl{\'e}mentin Boittiaux and Shizhe Chen and Vincent Lepetit},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=7WaRh4gCXp}
}
```

