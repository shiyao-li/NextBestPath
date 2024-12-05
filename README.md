# Instructions

1. Follow the instructions from MACARONS (https://github.com/Anttwo/MACARONS) to add the weights of macarons/depth
Note: We are using the perfect depth map and not the model from Macarons. However, I don't have enough time to modify my code to remove that part.
   
2. Create a conda env:
```python
conda env create -f environment.yml
conda activate macarons
```

3. Download the dataset from the below link:
[AiMDoom](https://drive.google.com/drive/folders/14IyZZw-HyXhWWfmcC_3xdhBS0lSHJ1jo?usp=sharing)

4. Download the NBP model weights from the same link.

Doom1 equals to Doom_Simple

Doom2 equals to Doom_Normal

Doom3 equals to Doom_Hard

Doom4 equals to Doom_Insane

Eg. If you are testing on the simple dataset, use the weights of doom1 and set the number of camera poses to 101.

Store these models in:
```python
./weights/navi/
```

5. Modify config file: test_via_navi_model.json

6. Run the script:
```python
test_navi_planning_2d.py
```
