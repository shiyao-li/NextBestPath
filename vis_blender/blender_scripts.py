import bpy
import json
from mathutils import Vector
import numpy as np
import blender_plots as bplt
import os

folder_path = "./nextbestpath" 
file_path = os.path.join(folder_path, "point_cloud.json")
with open(file_path, 'r') as f:
    data = json.load(f)
all_points = np.array(data['points'])
all_colors = np.array(data['colors'])
scatter = bplt.Scatter(all_points, color=all_colors, size=(0.4, 0.4, 0.4))
obj = bpy.data.objects.get("scatter")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.context.object.rotation_euler[0] = 1.5708
bpy.context.object.scale[0] = 0.069
bpy.context.object.scale[1] = 0.069
bpy.context.object.scale[2] = 0.069

bpy.context.object.location[0] = -0.131367
bpy.context.object.location[1] = 0.289053
bpy.context.object.location[2] = -0.08237


json_file_path = "trajectory.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)

# load the data of trajectory
X_cam_history = data["ashfall_of_the_infidel_3"]["0"]["X_cam_history"]
curve_object_name = 'TrajectoryCurveObject'

if curve_object_name in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[curve_object_name], do_unlink=True)

n = 4
sampled_points = X_cam_history[::n]

curve_data = bpy.data.curves.new(name='TrajectoryCurve', type='CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 12
curve_data.bevel_depth = 0.2

curve_object = bpy.data.objects.new('TrajectoryCurveObject', curve_data)
bpy.context.collection.objects.link(curve_object)

polyline = curve_data.splines.new('BEZIER')
polyline.bezier_points.add(len(sampled_points) - 1)

for i, point in enumerate(sampled_points):
    bp = polyline.bezier_points[i]
    bp.co = Vector(point)
    if i > 0:
        bp.handle_left_type = 'AUTO'
    if i < len(sampled_points) - 1:
        bp.handle_right_type = 'AUTO'

material = bpy.data.materials.new(name="TrajectoryMaterial")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links

for node in nodes:
    nodes.remove(node)

output_node = nodes.new(type='ShaderNodeOutputMaterial')
principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
gradient_texture = nodes.new(type='ShaderNodeTexGradient')
texture_coord = nodes.new(type='ShaderNodeTexCoord')
mapping = nodes.new(type='ShaderNodeMapping')
color_ramp = nodes.new(type='ShaderNodeValToRGB')

gradient_texture.gradient_type = 'LINEAR'

color_ramp.color_ramp.interpolation = 'LINEAR'
color_ramp.color_ramp.elements[0].position = 0.0
color_ramp.color_ramp.elements[0].color = (0, 0, 1, 1)
color_ramp.color_ramp.elements[1].position = 1.0
color_ramp.color_ramp.elements[1].color = (0, 1, 0, 1)

links.new(texture_coord.outputs['Generated'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], gradient_texture.inputs['Vector'])
links.new(gradient_texture.outputs['Color'], color_ramp.inputs['Fac'])
links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])
links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

curve_object.data.materials.append(material)

bpy.context.view_layer.objects.active = curve_object
curve_object.select_set(True)
bpy.context.object.rotation_euler[0] = 1.5708
bpy.context.object.scale[0] = 0.069
bpy.context.object.scale[1] = 0.069
bpy.context.object.scale[2] = 0.069

bpy.context.object.location[0] = -0.131367
bpy.context.object.location[1] = 0.289053
bpy.context.object.location[2] = -0.082370