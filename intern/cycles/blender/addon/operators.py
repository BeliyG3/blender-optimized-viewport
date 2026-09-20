# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, StringProperty

from bpy.app.translations import pgettext_tip as tip_


def _dlss_version_items(self, context):
    """The versions this driver provides for the feature the operator was given.

    The list itself, and the reason it has to outlive this call, are in `dlss_library`. `self` here
    is the operator's properties rather than the operator, and `context` can be None.
    """
    from . import dlss_library

    feature = getattr(self, "feature", "") or "dlssd"
    if feature not in dlss_library.FEATURES:
        feature = "dlssd"
    return dlss_library.version_enum_items(feature)


class CYCLES_OT_use_vulkan_next_launch(Operator):
    bl_idname = "cycles.use_vulkan_next_launch"
    bl_label = "Use Vulkan on Next Launch"
    bl_description = (
        "Save Vulkan as the GPU backend for the next Blender launch; "
        "Blender will not restart automatically"
    )

    def execute(self, context):
        system = context.preferences.system
        if not hasattr(system, "gpu_backend"):
            self.report({'ERROR'}, "This Blender build does not expose GPU backend preferences")
            return {'CANCELLED'}

        system.gpu_backend = 'VULKAN'
        bpy.ops.wm.save_userpref()
        self.report({'INFO'}, "Vulkan will be used after Blender is restarted")
        return {'FINISHED'}


class CYCLES_OT_install_dlss_library(Operator):
    bl_idname = "cycles.install_dlss_library"
    bl_label = "Install DLSS Library"
    bl_description = (
        "Copy the DLSS library the NVIDIA driver already installed into Blender's user folder, "
        "under the name NGX looks for. Nothing is downloaded"
    )

    feature: StringProperty(
        name="Feature",
        description="Which NGX feature to install: dlssd for Ray Reconstruction, dlssg for "
                    "Frame Generation",
        default="dlssd",
        options={'HIDDEN'},
    )

    version: EnumProperty(
        name="Version",
        description="Which driver-installed version to use. Newest first; a newer library is not "
                    "always the better one",
        items=_dlss_version_items,
        options={'HIDDEN'},
    )

    def execute(self, context):
        from . import dlss_library

        # A dynamic enum takes no default, so an unset property is the first item - the newest
        # version, which is what a plain button asking for no version in particular should get.
        version = "" if self.version == 'NONE' else self.version

        ok, message = dlss_library.install(self.feature, version)
        if not ok:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        self.report({'INFO'}, message)
        return {'FINISHED'}


class CYCLES_OT_remove_dlss_library(Operator):
    bl_idname = "cycles.remove_dlss_library"
    bl_label = "Remove DLSS Library"
    bl_description = (
        "Delete one installed DLSS library. The version this session is using cannot be deleted "
        "while it runs - restart Blender first"
    )

    feature: StringProperty(
        name="Feature",
        description="Which NGX feature to remove: dlssd for Ray Reconstruction, dlssg for "
                    "Frame Generation",
        default="dlssd",
        options={'HIDDEN'},
    )

    version: StringProperty(
        name="Version",
        description="Which installed version to delete",
        default="",
        options={'HIDDEN'},
    )

    def execute(self, context):
        from . import dlss_library

        ok, message = dlss_library.remove(self.feature, self.version)
        if not ok:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        self.report({'INFO'}, message)
        return {'FINISHED'}


class CYCLES_OT_use_shading_nodes(Operator):
    """Enable nodes on a light"""
    bl_idname = "cycles.use_shading_nodes"
    bl_label = "Use Nodes"

    @classmethod
    def poll(cls, context):
        return getattr(context, "light", False)

    def execute(self, context):
        if context.light:
            context.light.use_nodes = True

        return {'FINISHED'}


class CYCLES_OT_denoise_animation(Operator):
    "Denoise rendered animation sequence using current scene and view " \
        "layer settings. Requires denoising data passes and output to " \
        "OpenEXR multilayer files"
    bl_idname = "cycles.denoise_animation"
    bl_label = "Denoise Animation"

    input_filepath: StringProperty(
        name='Input Filepath',
        description='File path for image to denoise. If not specified, uses the render file path and frame range from the scene',
        default='',
        subtype='FILE_PATH')

    output_filepath: StringProperty(
        name='Output Filepath',
        description='If not specified, renders will be denoised in-place',
        default='',
        subtype='FILE_PATH')

    def execute(self, context):
        import os

        preferences = context.preferences
        scene = context.scene
        view_layer = context.view_layer

        in_filepath = self.input_filepath
        out_filepath = self.output_filepath

        in_filepaths = []
        out_filepaths = []

        if in_filepath != '':
            # Denoise a single file
            if out_filepath == '':
                out_filepath = in_filepath

            in_filepaths.append(in_filepath)
            out_filepaths.append(out_filepath)
        else:
            # Denoise animation sequence with expanded frames matching
            # Blender render output file naming.
            in_filepath = scene.render.filepath
            if out_filepath == '':
                out_filepath = in_filepath

            # Backup since we will overwrite the scene path temporarily
            original_filepath = scene.render.filepath

            for frame in range(scene.frame_start, scene.frame_end + 1):
                scene.render.filepath = in_filepath
                filepath = scene.render.frame_path(frame=frame)
                in_filepaths.append(filepath)

                if not os.path.isfile(filepath):
                    scene.render.filepath = original_filepath
                    err_msg = tip_("Frame '%s' not found, animation must be complete") % filepath
                    self.report({'ERROR'}, err_msg)
                    return {'CANCELLED'}

                scene.render.filepath = out_filepath
                filepath = scene.render.frame_path(frame=frame)
                out_filepaths.append(filepath)

            scene.render.filepath = original_filepath

        # Run denoiser
        # TODO: support cancel and progress reports.
        import _cycles
        try:
            _cycles.denoise(preferences.as_pointer(),
                            scene.as_pointer(),
                            view_layer.as_pointer(),
                            input=in_filepaths,
                            output=out_filepaths)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'FINISHED'}

        self.report({'INFO'}, "Denoising completed")
        return {'FINISHED'}


class CYCLES_OT_merge_images(Operator):
    "Combine OpenEXR multi-layer images rendered with different sample " \
        "ranges into one image with reduced noise"
    bl_idname = "cycles.merge_images"
    bl_label = "Merge Images"

    input_filepath1: StringProperty(
        name='Input Filepath',
        description='File path for image to merge',
        default='',
        subtype='FILE_PATH')

    input_filepath2: StringProperty(
        name='Input Filepath',
        description='File path for image to merge',
        default='',
        subtype='FILE_PATH')

    output_filepath: StringProperty(
        name='Output Filepath',
        description='File path for merged image',
        default='',
        subtype='FILE_PATH')

    def execute(self, context):
        in_filepaths = [self.input_filepath1, self.input_filepath2]
        out_filepath = self.output_filepath

        import _cycles
        try:
            _cycles.merge(input=in_filepaths, output=out_filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'FINISHED'}

        return {'FINISHED'}


classes = (
    CYCLES_OT_use_vulkan_next_launch,
    CYCLES_OT_install_dlss_library,
    CYCLES_OT_remove_dlss_library,
    CYCLES_OT_use_shading_nodes,
    CYCLES_OT_denoise_animation,
    CYCLES_OT_merge_images
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)


def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        unregister_class(cls)
