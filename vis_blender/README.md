# Blender Visualization Scripts

This directory contains scripts for visualizing **NextBestPath** results in [Blender](https://www.blender.org/), including point cloud rendering and camera trajectory display.

## What the script does

`blender_scripts.py` loads and visualizes two things:

1. **Point cloud** — reads `point_cloud.json` (XYZ + RGB) and renders it as a scatter plot using `blender_plots`.
2. **Camera trajectory** — reads `trajectory.json` and draws it as a smooth 3D Bezier curve with a blue-to-green gradient material.

## Requirements

- **Blender >= 3.6** (recommended)
- **blender-plots** addon: [https://github.com/Linusnie/blender-plots](https://github.com/Linusnie/blender-plots)

### Installing blender-plots

The easiest way is to install it as a Blender addon:

1. Download the repository as a `.zip` from GitHub.
2. In Blender, go to `Edit > Preferences > Add-ons > Install...`.
3. Select the downloaded `.zip` and enable the addon.

## How to run the script in Blender

1. Open Blender.
2. Switch one of the editor panels to **Scripting** (top menu or editor type selector).
3. Click **Open** and select `blender_scripts.py`, or paste the code directly into the text editor.
4. Make sure the input files are accessible:
   - `./nextbestpath/point_cloud.json`
   - `./trajectory.json`
5. Click **Run Script** (or press `Alt + P`).

## Setting up the camera

You can freely adjust the active camera to find a good viewing angle before rendering.

- Select the camera object, press `Numpad 0` to enter camera view, then use `G` / `R` to move/rotate it.
- Or navigate to a view you like and use `View > Align View > Align Active Camera to View`.
- For a beginner-friendly walkthrough, refer to this video: [https://www.youtube.com/watch?v=exHZ9BnBQGw](https://www.youtube.com/watch?v=exHZ9BnBQGw). There are also many tutorials on Bilibili.

## Removing ceiling points

Indoor point clouds often include ceiling points that obstruct the view. You can remove them in two ways:

**Option A — manually in Blender:** select the scatter object, enter Edit Mode, box-select the top points, and delete them.

**Option B — filter by Z threshold in the script:** add the following lines right after `all_points` and `all_colors` are loaded:

```python
z_threshold = 2.5  # adjust to your scene's ceiling height
mask = all_points[:, 2] < z_threshold
all_points = all_points[mask]
all_colors = all_colors[mask]
```

This keeps only points whose Z coordinate is below `z_threshold`, effectively removing the ceiling before the scatter plot is created.
