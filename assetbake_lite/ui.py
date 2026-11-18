# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar tab (3D Viewport > N panel), Object Mode."""
import json
import textwrap

from bpy.types import Panel

from . import const, ops

ICONS = {"INFO": "INFO", "WARNING": "ERROR", "ERROR": "CANCEL"}


def wrapped(layout, context, text, icon="BLANK1", inset=70):
    """A label cuts long text with "..."; this breaks it into lines that fit the sidebar. Icon on the first line.
    `inset` = pixels (at scale 1) the layout loses to margins, icon and boxes."""
    scale = context.preferences.system.ui_scale
    chars = max(16, int((context.region.width - inset * scale) / (6.5 * scale)))
    for n, line in enumerate(textwrap.wrap(text, chars)):
        layout.label(text=line, icon=icon if n == 0 else "BLANK1")


class ASSETBAKELITE_PT_panel(Panel):
    bl_label = "%s %s" % (const.ADDON_NAME, const.VERSION)       # the version comes from blender_manifest.toml
    bl_idname = "ASSETBAKELITE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = const.TAB

    def draw(self, context):
        p = ops.settings(context)
        layout = self.layout
        running = ops.is_running()
        object_mode = context.mode == "OBJECT"
        col = layout.column(align=True)
        col.enabled = not running and object_mode

        quads_off, quads = "", False
        box = col.box()
        box.label(text="Face budget (quads + triangles)" if quads else "Triangle budget", icon="MESH_DATA")
        box.prop(p, "budget_preset", text="")
        if quads and p.budget() < const.QUAD_MIN_BUDGET:
            wrapped(box, context, "Quad-dominant needs at least {:,} faces.".format(const.QUAD_MIN_BUDGET), "CANCEL",
                    inset=100)
        elif quads and p.budget() < const.QUAD_ADVISED_BUDGET:
            wrapped(box, context, "Below {:,} faces the quad share drops and a fallback to Game is more "
                                  "likely.".format(const.QUAD_ADVISED_BUDGET), "ERROR", inset=100)

        box = col.box()
        box.label(text="Textures", icon="TEXTURE")
        box.row(align=True).prop(p, "texture_size", expand=True)
        grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        grid.label(text="Normal", icon="CHECKBOX_HLT")
        for _, prop, _, _ in ops.MAP_SETTINGS:
            grid.prop(p, prop)

        box = col.box()
        box.label(text="Options", icon="OUTLINER_OB_GROUP_INSTANCE")
        box.prop(p, "remove_hidden")

        # progress and Cancel are shown in every mode: the job keeps running when the user tabs into Edit Mode
        if running:
            job = ops.current_job()
            if job.state == "ready":
                wrapped(layout, context, ops.WAITING, "INFO")
            else:
                layout.progress(factor=max(0.0, min(1.0, job.fraction)), type="BAR", text=job.text)
            row = layout.row()
            row.scale_y = 1.4
            row.operator(const.OPERATOR + ".cancel", icon="CANCEL")
            return
        if not object_mode:
            layout.label(text="Go to Object Mode to make an asset", icon="INFO")
            return
        objs = ops.sources(context)
        tris = ops.triangle_count(objs)
        info = layout.column(align=True)
        if not objs:
            if ops.only_assets_selected(context):
                wrapped(info, context, "This is an asset made by this add-on, it is not used as a source. To make "
                                       "another version, select the original objects:", "INFO")
                row = info.row()
                row.scale_y = 1.4
                row.operator(const.OPERATOR + ".select_source", icon="RESTRICT_SELECT_OFF").last = False
            else:
                info.label(text="Select one or more mesh objects", icon="INFO")
        else:
            info.label(text="Selected: %d object%s, %s triangles" % (
                len(objs), "" if len(objs) == 1 else "s", "{:,}".format(tris)))
            if tris > const.REFUSE_TRIS:
                wrapped(info, context, "Too large: the limit is {:,} triangles. Decimate the source "
                                       "first.".format(const.REFUSE_TRIS), "CANCEL")
            elif tris > const.WARN_TRIS:
                wrapped(info, context, "Large input: this takes several minutes and several GB of memory.", "ERROR")
            if quads and tris <= const.REFUSE_TRIS and p.budget() >= const.QUAD_MIN_BUDGET:
                # what to expect, before the wait: what will happen when that is known without measuring, else a
                # quick look at the wall thickness of the selection
                level, text, measure = ops.quad_outlook(tris, p.budget(), "SELECTION")
                if measure:
                    known, thin = ops.quad_estimate_cached(objs, p.budget(), ops.estimate_allowed(context))
                    if known:
                        level, text = ops.quad_expectation(thin, p.budget())
                    elif ops.estimate_allowed(context):
                        text = "Estimating the quad share..."
                if text:
                    wrapped(info, context, text, dict(INFO="INFO", WARNING="ERROR", ERROR="CANCEL")[level])
        row = layout.row()
        row.scale_y = 1.6
        row.enabled = bool(objs) and tris <= const.REFUSE_TRIS and not (quads and p.budget() < const.QUAD_MIN_BUDGET)
        row.operator(const.OPERATOR + ".make", icon="MOD_DECIM")


class ASSETBAKELITE_PT_result(Panel):
    bl_label = "Result  ·  %s v%s" % (const.ADDON_NAME, const.VERSION)
    bl_idname = "ASSETBAKELITE_PT_zresult"   # panels of a tab are listed alphabetically by idname: z = last
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = const.TAB

    def draw(self, context):
        p = ops.settings(context)
        layout = self.layout
        try:
            rep = json.loads(p.report) if p.report else None
        except ValueError:
            rep = None
        if not isinstance(rep, dict):
            rep = None
        # the report is saved with the file and may come from another version of the add-on: read it with .get
        if ops.is_running():
            waiting = ops.current_job().state == "ready"
            layout.label(text="Finished, waiting for Object Mode" if waiting else "Working in the background...",
                         icon="SORTTIME")
        elif not rep:
            layout.label(text="No result yet", icon="INFO")
        elif not rep.get("ok"):
            wrapped(layout, context, rep.get("error", "The job failed."), "INFO" if rep.get("cancelled") else "CANCEL")
            if rep.get("log"):
                wrapped(layout, context, "Log file for support: " + rep["log"])
                layout.operator(const.OPERATOR + ".copy_log_path", icon="COPYDOWN")
        else:
            for message in rep.get("messages", []):
                wrapped(layout, context, str(message[1]), ICONS.get(message[0], "INFO"))
            for asset in rep.get("assets", []):
                box = layout.box()
                box.label(text=asset.get("name", "asset"), icon="CHECKMARK")
                col = box.column(align=True)
                for line in asset.get("lines", []):
                    wrapped(col, context, str(line), inset=100)
                for message in asset.get("messages", []):
                    wrapped(col, context, str(message[1]), ICONS.get(message[0], "INFO"), inset=100)
            wrapped(layout, context, "The source is hidden, not changed. Total time: %.0f s." % rep.get("seconds", 0))
            wrapped(layout, context, "For another version (other budget or textures), get the source back:")
            layout.operator(const.OPERATOR + ".select_source", icon="RESTRICT_SELECT_OFF").last = True

        if context.mode != "OBJECT":
            return
        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.prop(p, "export_textures")
        row = box.row()
        row.scale_y = 1.4
        row.operator(const.OPERATOR + ".export_glb", icon="EXPORT")


class ASSETBAKELITE_PT_full(Panel):
    bl_label = "Full version"
    bl_idname = "ASSETBAKELITE_PT_zzfull"   # after the result panel
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = const.TAB

    def draw(self, context):
        wrapped(self.layout, context, const.FULL_FEATURES, "INFO")
        self.layout.operator("wm.url_open", text="Get the full version", icon="URL").url = const.FULL_URL


classes = (ASSETBAKELITE_PT_panel, ASSETBAKELITE_PT_result, ASSETBAKELITE_PT_full)
