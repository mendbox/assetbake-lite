# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 6: the material of the asset.

The node layout is the one Blender's glTF exporter understands, so the asset exports correctly both with the
add-on's Export button and with File > Export > glTF: one ORM image through Separate Color (G = roughness,
B = metallic), occlusion (R) into the "glTF Material Output" group, a Normal Map node, alpha as a cut-out
through a Round node (alphaMode MASK).
"""
import importlib

import bpy

OCCLUSION_GROUP = "glTF Material Output"      # the name the glTF exporter looks for
NODE_PREFIX = "asset_"                        # image nodes are named asset_<kind>: the exporter code finds them so


def _gltf_group_factory():
    """The glTF add-on's own function that builds its settings group. The importer reuses any group with this
    name and expects all of its sockets, so a hand-made group with only "Occlusion" breaks later glTF imports."""
    for module in ("io_scene_gltf2.blender.com.material_helpers",                 # Blender 4.5, 5.x
                   "io_scene_gltf2.blender.com.gltf2_blender_material_helpers"):   # Blender 4.2
        try:
            return importlib.import_module(module).create_settings_group
        except (ImportError, AttributeError):
            continue
    return None


def _occlusion_group():
    group = bpy.data.node_groups.get(OCCLUSION_GROUP)
    if group is None:
        factory = _gltf_group_factory()
        if factory is not None:
            group = factory(OCCLUSION_GROUP)
        else:
            group = bpy.data.node_groups.new(OCCLUSION_GROUP, "ShaderNodeTree")
            group.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    return group


def image_node(mat, kind):
    return mat.node_tree.nodes.get(NODE_PREFIX + kind) if mat and mat.node_tree else None


def build(name, images, constants, emission_strength=1.0, has_alpha=False, has_occlusion=False):
    """images: {"normal", "basecolor", "orm", "emission"} -> image (all but "normal" optional)."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None or not mat.node_tree.nodes:
        mat.use_nodes = True                      # deprecated from Blender 5.0, where the tree is always there
    tree = mat.node_tree
    shader = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value = constants["roughness"]
    shader.inputs["Metallic"].default_value = constants["metallic"]
    shader.inputs["Base Color"].default_value = tuple(constants["basecolor"]) + (1.0,)
    link = tree.links.new
    x, y = shader.location.x - 900, shader.location.y + 300

    def texture(kind, label):
        nonlocal y
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = images[kind]
        node.name = NODE_PREFIX + kind
        node.label = label
        node.location = (x, y)
        y -= 300
        return node

    if "basecolor" in images:
        base = texture("basecolor", "Base Colour")
        link(base.outputs["Color"], shader.inputs["Base Color"])
        if has_alpha:
            cut = tree.nodes.new("ShaderNodeMath")
            cut.operation = "ROUND"
            cut.location = (x + 400, base.location.y - 120)
            link(base.outputs["Alpha"], cut.inputs[0])
            link(cut.outputs[0], shader.inputs["Alpha"])
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = "DITHERED"
            if hasattr(mat, "blend_method"):
                mat.blend_method = "CLIP"
    if "orm" in images:
        orm = texture("orm", "Occlusion / Roughness / Metallic")
        split = tree.nodes.new("ShaderNodeSeparateColor")
        split.location = (x + 400, orm.location.y)
        link(orm.outputs["Color"], split.inputs[0])
        link(split.outputs["Green"], shader.inputs["Roughness"])
        link(split.outputs["Blue"], shader.inputs["Metallic"])
        if has_occlusion:
            group = tree.nodes.new("ShaderNodeGroup")
            group.node_tree = _occlusion_group()
            group.location = (x + 650, orm.location.y + 120)
            link(split.outputs["Red"], group.inputs["Occlusion"])
    if "emission" in images:
        glow = texture("emission", "Emission")
        link(glow.outputs["Color"], shader.inputs["Emission Color"])
        shader.inputs["Emission Strength"].default_value = emission_strength
    normal = texture("normal", "Normal (OpenGL)")
    bump = tree.nodes.new("ShaderNodeNormalMap")
    bump.location = (x + 400, normal.location.y)
    link(normal.outputs["Color"], bump.inputs["Color"])
    link(bump.outputs["Normal"], shader.inputs["Normal"])
    return mat
