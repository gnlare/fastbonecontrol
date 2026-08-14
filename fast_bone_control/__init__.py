bl_info = {
    "name": "Fast Bone Control",
    "author": "Prototype",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > N-Panel > Fast Bone",
    "description": (
        "Fast direct bone transform controls with right-click mode cycling, "
        "drag transforms, Shift snapping, and mode gizmos."
    ),
    "category": "Rigging",
}

import math
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from bpy.types import Operator, Panel, WorkSpaceTool
from bpy.props import EnumProperty, BoolProperty, FloatProperty

MODES = ["TRANSLATE", "ROTATE", "SCALE"]
MODE_LABELS = {"TRANSLATE": "Move (G)", "ROTATE": "Rotate (R)", "SCALE": "Scale (S)"}
MODE_COLORS = {
    "TRANSLATE": (0.25, 0.9, 0.45, 0.95),   # green
    "ROTATE": (0.3, 0.6, 1.0, 0.95),        # blue
    "SCALE": (1.0, 0.6, 0.15, 0.95),        # orange
}

TOOL_IDNAME = "spine.tweak_tool"
DEFAULT_TWEAK_ID = "builtin.select"


def _tag_redraw_view3d(context):
    screen = context.screen
    if not screen:
        return
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _find_view3d_window_region(context):
    win = context.window
    screen = context.screen or (win.screen if win else None)
    if not screen:
        return None, None, None
    for area in screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    return win, area, region
    return None, None, None


def _sync_active_tool(context):
    win, area, region = _find_view3d_window_region(context)
    if not (win and area and region):
        return
    target = TOOL_IDNAME if context.window_manager.spine_mode_enabled else DEFAULT_TWEAK_ID
    with context.temp_override(window=win, area=area, region=region):
        try:
            bpy.ops.wm.tool_set_by_id(name=target, space_type="VIEW_3D")
        except RuntimeError:
            pass


def _get_enabled_modes(wm):
    modes = []
    if wm.spine_mode_use_translate:
        modes.append("TRANSLATE")
    if wm.spine_mode_use_rotate:
        modes.append("ROTATE")
    if wm.spine_mode_use_scale:
        modes.append("SCALE")
    return modes


def _ensure_valid_current_mode(wm):
    enabled = _get_enabled_modes(wm)
    if enabled and wm.spine_bone_mode not in enabled:
        wm.spine_bone_mode = enabled[0]


def _enforce_at_least_one_checked(wm, changed_attr, context):
    if not (wm.spine_mode_use_translate or wm.spine_mode_use_rotate or wm.spine_mode_use_scale):
        # At least one mode must stay enabled; restore the option that was just disabled.
        setattr(wm, changed_attr, True)
        return
    _ensure_valid_current_mode(wm)
    _tag_redraw_view3d(context)


def _on_mode_update(self, context):
    _tag_redraw_view3d(context)


def _on_enabled_update(self, context):
    _tag_redraw_view3d(context)
    _sync_active_tool(context)


def _on_use_translate_update(self, context):
    _enforce_at_least_one_checked(self, "spine_mode_use_translate", context)


def _on_use_rotate_update(self, context):
    _enforce_at_least_one_checked(self, "spine_mode_use_rotate", context)


def _on_use_scale_update(self, context):
    _enforce_at_least_one_checked(self, "spine_mode_use_scale", context)


# ---------------------------------------------------------------------------
# Right-click: cycle only through enabled transform modes
# ---------------------------------------------------------------------------
class POSE_OT_spine_cycle_mode(Operator):
    """Cycle only through enabled bone transform modes with right-click."""

    bl_idname = "pose.spine_cycle_mode"
    bl_label = "Cycle Fast Bone Mode"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and context.selected_pose_bones

    def execute(self, context):
        wm = context.window_manager
        enabled = _get_enabled_modes(wm)
        if not enabled:
            return {"CANCELLED"}

        if wm.spine_bone_mode not in enabled:
            wm.spine_bone_mode = enabled[0]
        else:
            idx = enabled.index(wm.spine_bone_mode)
            wm.spine_bone_mode = enabled[(idx + 1) % len(enabled)]

        self.report({"INFO"}, f"Fast Bone mode: {MODE_LABELS[wm.spine_bone_mode]}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Left-click drag: transform selected bones by dragging anywhere in the viewport
# ---------------------------------------------------------------------------
class POSE_OT_spine_drag_transform(Operator):
    """Transform selected bones in the current mode by dragging anywhere in the viewport."""

    bl_idname = "pose.spine_drag_transform"
    bl_label = "Fast Bone Drag Transform"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE"

    def invoke(self, context, event):
        if not context.selected_pose_bones:
            return bpy.ops.view3d.select_box("INVOKE_DEFAULT")

        wm = context.window_manager
        _ensure_valid_current_mode(wm)
        mode = wm.spine_bone_mode
        if mode == "TRANSLATE":
            return bpy.ops.transform.translate("INVOKE_DEFAULT")
        elif mode == "ROTATE":
            return bpy.ops.transform.rotate("INVOKE_DEFAULT")
        else:
            return bpy.ops.transform.resize("INVOKE_DEFAULT")


# ---------------------------------------------------------------------------
# Shift + left-click drag: snap transforms using user-defined increments
# - Move: snap the view-plane movement direction by the configured angle
# - Rotate: snap by the configured angle
# - Scale: snap by the configured scale increment
# Normal dragging continues to use Blender's built-in transform operators.
# ---------------------------------------------------------------------------
class POSE_OT_spine_shift_snap_transform(Operator):
    """Snap move, rotate, or scale by configured increments while Shift-dragging."""

    bl_idname = "pose.spine_shift_snap_transform"
    bl_label = "Fast Bone Shift Snap"
    bl_options = {"REGISTER", "UNDO", "BLOCKING"}

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(context.selected_pose_bones)

    @staticmethod
    def _bone_depth(pbone):
        depth = 0
        parent = pbone.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        return depth

    def _selected_bones(self, context):
        # Applying parent-to-child keeps simultaneous parent/child selections stable.
        bones = list(context.selected_pose_bones or [])
        bones.sort(key=self._bone_depth)
        return bones

    def _restore_initial(self):
        if self._obj is None:
            return
        # Restore parents before children so world/pose matrices return correctly.
        for pbone in self._bones:
            matrix = self._initial_matrices.get(pbone.name)
            if matrix is not None:
                pbone.matrix = matrix.copy()

    def _pivot_armature(self, pbone=None):
        if pbone is not None:
            return self._initial_heads[pbone.name].copy()
        return self._pivot_arm.copy()

    def _apply_translation(self, context, event):
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return

        raw_delta = Vector((
            event.mouse_region_x - self._start_mouse.x,
            event.mouse_region_y - self._start_mouse.y,
        ))
        if raw_delta.length < 1e-8:
            self._restore_initial()
            return

        # The Shift-move direction is determined once near the start of the drag.
        # The angle stays locked until mouse release and is not recalculated during the drag.
        if self._locked_move_angle is None:
            # Ignore tiny initial mouse jitter before locking the direction.
            if raw_delta.length < 3.0:
                return
            step_deg = max(0.1, context.window_manager.spine_snap_move_angle_deg)
            step = math.radians(step_deg)
            raw_angle = math.atan2(raw_delta.y, raw_delta.x)
            self._locked_move_angle = round(raw_angle / step) * step

        direction = Vector((
            math.cos(self._locked_move_angle),
            math.sin(self._locked_move_angle),
        ))

        # Project the current mouse displacement onto the initially locked line.
        # Preserve the sign so movement can pass the start point in the opposite direction on the same line.
        distance = raw_delta.dot(direction)
        snapped_mouse = self._start_mouse + direction * distance

        start_world = view3d_utils.region_2d_to_location_3d(
            region, rv3d, self._start_mouse, self._pivot_world
        )
        end_world = view3d_utils.region_2d_to_location_3d(
            region, rv3d, snapped_mouse, self._pivot_world
        )
        delta_world = end_world - start_world
        delta_arm = self._obj.matrix_world.inverted_safe().to_3x3() @ delta_world
        tmat = Matrix.Translation(delta_arm)

        self._restore_initial()
        for pbone in self._bones:
            pbone.matrix = tmat @ self._initial_matrices[pbone.name]

    def _screen_angle(self, mouse):
        v = mouse - self._pivot_screen
        if v.length < 1e-8:
            return 0.0
        return math.atan2(v.y, v.x)

    @staticmethod
    def _wrap_pi(value):
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _apply_rotation(self, context, event):
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        raw = self._wrap_pi(self._screen_angle(mouse) - self._start_screen_angle)
        step_deg = max(0.1, context.window_manager.spine_snap_rotate_angle_deg)
        step = math.radians(step_deg)
        snapped = round(raw / step) * step

        # Use the opposite sign of the view axis to match the visible rotation direction.
        axis_world = view3d_utils.region_2d_to_vector_3d(
            context.region, context.region_data, self._pivot_screen
        ).normalized()
        axis_arm = (self._obj.matrix_world.inverted_safe().to_3x3() @ axis_world).normalized()
        rmat = Matrix.Rotation(-snapped, 4, axis_arm)

        self._restore_initial()
        individual = context.scene.tool_settings.transform_pivot_point == 'INDIVIDUAL_ORIGINS'
        for pbone in self._bones:
            pivot = self._pivot_armature(pbone if individual else None)
            around = Matrix.Translation(pivot) @ rmat @ Matrix.Translation(-pivot)
            pbone.matrix = around @ self._initial_matrices[pbone.name]

    def _apply_scale(self, context, event):
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        radius = (mouse - self._pivot_screen).length
        if self._start_radius < 1e-6:
            factor = 1.0
        else:
            factor = radius / self._start_radius

        step = max(0.001, context.window_manager.spine_snap_scale_step)
        # Snap around 1.0, e.g. with 0.1 steps: 0.9, 1.0, 1.1, ...
        factor = 1.0 + round((factor - 1.0) / step) * step
        factor = max(0.001, factor)
        smat = Matrix.Diagonal((factor, factor, factor, 1.0))

        self._restore_initial()
        individual = context.scene.tool_settings.transform_pivot_point == 'INDIVIDUAL_ORIGINS'
        for pbone in self._bones:
            pivot = self._pivot_armature(pbone if individual else None)
            around = Matrix.Translation(pivot) @ smat @ Matrix.Translation(-pivot)
            pbone.matrix = around @ self._initial_matrices[pbone.name]

    def invoke(self, context, event):
        if not context.selected_pose_bones:
            return {"CANCELLED"}
        if context.region is None or context.region_data is None:
            return {"CANCELLED"}

        wm = context.window_manager
        _ensure_valid_current_mode(wm)
        self._mode = wm.spine_bone_mode
        self._obj = context.active_object
        self._bones = self._selected_bones(context)
        self._initial_matrices = {b.name: b.matrix.copy() for b in self._bones}
        self._initial_heads = {b.name: b.head.copy() for b in self._bones}

        if not self._bones or self._obj is None:
            return {"CANCELLED"}

        # Default pivot is the center of selected bone heads; Individual Origins uses each bone head separately.
        self._pivot_arm = sum((self._initial_heads[b.name] for b in self._bones), Vector((0, 0, 0))) / len(self._bones)
        self._pivot_world = self._obj.matrix_world @ self._pivot_arm
        pivot2d = view3d_utils.location_3d_to_region_2d(
            context.region, context.region_data, self._pivot_world
        )
        if pivot2d is None:
            return {"CANCELLED"}

        self._pivot_screen = Vector(pivot2d)
        self._start_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self._start_screen_angle = self._screen_angle(self._start_mouse)
        self._start_radius = (self._start_mouse - self._pivot_screen).length
        self._locked_move_angle = None

        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._restore_initial()
            context.window.cursor_modal_restore()
            _tag_redraw_view3d(context)
            return {"CANCELLED"}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            context.window.cursor_modal_restore()
            _tag_redraw_view3d(context)
            return {"FINISHED"}

        if event.type == 'MOUSEMOVE':
            if self._mode == 'TRANSLATE':
                self._apply_translation(context, event)
            elif self._mode == 'ROTATE':
                self._apply_rotation(context, event)
            else:
                self._apply_scale(context, event)
            _tag_redraw_view3d(context)

        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# Custom tool automatically activated while Fast Bone Control is active
# ---------------------------------------------------------------------------
class SpineTweakTool(WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "POSE"
    bl_idname = TOOL_IDNAME
    bl_label = "Fast Bone Tweak"
    bl_description = (
        "Fast direct bone selection and transform tool\n"
        "Click: Select (click empty space to deselect)\n"
        "Drag: Transform using the current mode (Move/Rotate/Scale)\n"
        "Shift + Drag: Snap using configured increments\n"
        "Right-click: Cycle included modes only"
    )
    bl_icon = "ops.transform.transform"
    bl_widget = None
    bl_keymap = (
        (POSE_OT_spine_shift_snap_transform.bl_idname, {"type": "LEFTMOUSE", "value": "CLICK_DRAG", "shift": True}, {"properties": []}),
        (POSE_OT_spine_drag_transform.bl_idname, {"type": "LEFTMOUSE", "value": "CLICK_DRAG"}, {"properties": []}),
        (POSE_OT_spine_cycle_mode.bl_idname, {"type": "RIGHTMOUSE", "value": "PRESS"}, {"properties": []}),
            ("view3d.fast_bone_shortcut_activate", {"type": "ONE", "value": "PRESS"}, {"properties": [("slot", 1)]}),
        ("view3d.fast_bone_shortcut_activate", {"type": "TWO", "value": "PRESS"}, {"properties": [("slot", 2)]}),
        ("view3d.fast_bone_shortcut_activate", {"type": "THREE", "value": "PRESS"}, {"properties": [("slot", 3)]}),
        ("view3d.fast_bone_shortcut_activate", {"type": "FOUR", "value": "PRESS"}, {"properties": [("slot", 4)]}),
        ("view3d.fast_bone_shortcut_activate", {"type": "FIVE", "value": "PRESS"}, {"properties": [("slot", 5)]}),
    )

    def draw_settings(context, layout, tool):
        # The current transform mode is changed only by right-click cycling and is not shown in tool settings.
        pass


# ---------------------------------------------------------------------------
# Per-mode gizmo (screen-space icon)
# ---------------------------------------------------------------------------
def _lines_cross_arrow(x, y, s):
    a = s * 0.35
    return [
        (x - s, y), (x + s, y),
        (x, y - s), (x, y + s),
        (x + s, y), (x + s - a, y + a * 0.6),
        (x + s, y), (x + s - a, y - a * 0.6),
        (x - s, y), (x - s + a, y + a * 0.6),
        (x - s, y), (x - s + a, y - a * 0.6),
        (x, y + s), (x + a * 0.6, y + s - a),
        (x, y + s), (x - a * 0.6, y + s - a),
        (x, y - s), (x + a * 0.6, y - s + a),
        (x, y - s), (x - a * 0.6, y - s + a),
    ]


def _lines_circle(x, y, s, segments=28):
    verts = []
    for i in range(segments):
        a1 = 2 * math.pi * i / segments
        a2 = 2 * math.pi * (i + 1) / segments
        verts.append((x + s * math.cos(a1), y + s * math.sin(a1)))
        verts.append((x + s * math.cos(a2), y + s * math.sin(a2)))
    tip = (x + s, y)
    verts += [
        tip, (tip[0] - s * 0.25, tip[1] + s * 0.2),
        tip, (tip[0] - s * 0.05, tip[1] - s * 0.25),
    ]
    return verts


def _lines_square(x, y, s):
    p1, p2, p3, p4 = (x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)
    verts = [p1, p2, p2, p3, p3, p4, p4, p1]
    h = s * 0.3
    for cx, cy, sx, sy in [
        (x - s, y - s, 1, 1), (x + s, y - s, -1, 1),
        (x + s, y + s, -1, -1), (x - s, y + s, 1, -1),
    ]:
        verts += [(cx, cy), (cx + h * sx, cy), (cx, cy), (cx, cy + h * sy)]
    return verts


def _get_active_bone_world_head(context):
    obj = context.active_object
    if obj is None or obj.type != "ARMATURE":
        return None
    pbone = context.active_pose_bone
    if pbone is None:
        return None
    return obj.matrix_world @ pbone.head


def draw_spine_mode_gizmo():
    context = bpy.context
    if context.mode != "POSE":
        return
    wm = context.window_manager
    if not wm.spine_mode_enabled:
        return
    if not context.selected_pose_bones:
        return

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    world_pos = _get_active_bone_world_head(context)
    if world_pos is None:
        return

    co2d = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
    if co2d is None:
        return

    mode = wm.spine_bone_mode
    if mode not in _get_enabled_modes(wm):
        return
    color = MODE_COLORS[mode]
    x, y = co2d
    size = 20

    if mode == "TRANSLATE":
        verts = _lines_cross_arrow(x, y, size)
    elif mode == "ROTATE":
        verts = _lines_circle(x, y, size)
    else:
        verts = _lines_square(x, y, size)

    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, "LINES", {"pos": verts})
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


# ---------------------------------------------------------------------------
# Fast Bone Control enter / exit
# ---------------------------------------------------------------------------
def _selected_armature_for_pose(context):
    active = context.active_object
    if active is not None and active.type == "ARMATURE" and active.select_get():
        return active

    for obj in context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
    return None


class VIEW3D_OT_fast_bone_mode_toggle(Operator):
    """Enter or exit Fast Bone Control. Entering requires a selected Armature."""

    bl_idname = "view3d.fast_bone_mode_toggle"
    bl_label = "Toggle Fast Bone Control"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        wm = context.window_manager

        if wm.spine_mode_enabled:
            # Exit Fast Bone Control and always return to Object Mode.
            wm.spine_mode_enabled = False

            if context.mode != "OBJECT":
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except RuntimeError:
                    self.report({"ERROR"}, "Unable to return to Object Mode.")
                    return {"CANCELLED"}

            _sync_active_tool(context)
            _tag_redraw_view3d(context)
            self.report({"INFO"}, "Exited Fast Bone Control and returned to Object Mode.")
            return {"FINISHED"}

        armature = _selected_armature_for_pose(context)
        if armature is None:
            self.report({"ERROR"}, "Select an Armature before entering Fast Bone Control.")
            return {"CANCELLED"}

        # Leave the current mode first so the selected Armature can become active.
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                self.report({"ERROR"}, "Unable to switch to Object Mode from the current state.")
                return {"CANCELLED"}

        try:
            bpy.ops.object.select_all(action="DESELECT")
            armature.hide_set(False)
            armature.select_set(True)
            context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode="POSE")
        except RuntimeError:
            self.report({"ERROR"}, f"Unable to enter Pose Mode for '{armature.name}'.")
            return {"CANCELLED"}

        wm.spine_mode_enabled = True
        _ensure_valid_current_mode(wm)
        _sync_active_tool(context)
        _tag_redraw_view3d(context)
        self.report({"INFO"}, f"Entered Fast Bone Control: {armature.name}")
        return {"FINISHED"}



# ---------------------------------------------------------------------------
# Armature shortcuts
# ---------------------------------------------------------------------------
_shortcut_cycle_state = {}

def _shortcut_prop_name(slot, button):
    return f"fast_bone_shortcut_{slot}_{button}"

def _shortcut_target_name(armature_obj, slot, button):
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return ""
    return armature_obj.data.get(_shortcut_prop_name(slot, button), "")

class VIEW3D_OT_fast_bone_shortcut_pick_target(Operator):
    bl_idname = "view3d.fast_bone_shortcut_pick_target"
    bl_label = "Pick Armature Shortcut Target"
    bl_options = {"REGISTER"}

    slot: bpy.props.IntProperty(min=1, max=5, default=1)
    button: bpy.props.IntProperty(min=1, max=3, default=1)

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return (
            context.area is not None and context.area.type == 'VIEW_3D'
            and context.mode == 'POSE' and wm.spine_mode_enabled
            and context.active_object is not None
            and context.active_object.type == 'ARMATURE'
        )

    def invoke(self, context, event):
        self._area = context.area
        self._armature = context.active_object
        _win, _area, region = _find_view3d_window_region(context)
        self._region = region
        if region is None:
            return {'CANCELLED'}
        context.window.cursor_modal_set('EYEDROPPER')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass
        _tag_redraw_view3d(context)

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._finish(context)
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if context.area != self._area:
                return {'RUNNING_MODAL'}

            rx = event.mouse_x - self._region.x
            ry = event.mouse_y - self._region.y
            if rx < 0 or ry < 0 or rx >= self._region.width or ry >= self._region.height:
                return {'RUNNING_MODAL'}

            arm = self._armature
            if arm is None or arm.type != 'ARMATURE' or context.mode != 'POSE':
                self._finish(context)
                return {'CANCELLED'}

            # Clear pose selection first so an empty click can be detected reliably.
            for pb in arm.pose.bones:
                try:
                    pb.select = False
                except Exception:
                    pass

            try:
                with context.temp_override(window=context.window, area=self._area, region=self._region):
                    bpy.ops.view3d.select(
                        location=(int(rx), int(ry)),
                        extend=False, deselect=False, toggle=False,
                        center=False, enumerate=False, object=False,
                    )
            except (RuntimeError, TypeError):
                try:
                    with context.temp_override(window=context.window, area=self._area, region=self._region):
                        bpy.ops.view3d.select(location=(int(rx), int(ry)), extend=False)
                except RuntimeError:
                    return {'RUNNING_MODAL'}

            selected = []
            for pb in arm.pose.bones:
                try:
                    if pb.select:
                        selected.append(pb)
                except Exception:
                    pass

            prop = _shortcut_prop_name(self.slot, self.button)
            try:
                window_key = context.window.as_pointer()
            except Exception:
                window_key = 0

            if not selected or context.active_object != arm:
                if prop in arm.data:
                    del arm.data[prop]
                _shortcut_cycle_state.pop((window_key, self.slot), None)
                self._finish(context)
                return {'FINISHED'}

            target = context.active_pose_bone
            if target not in selected:
                target = selected[-1]
            arm.data[prop] = target.name
            _shortcut_cycle_state.pop((window_key, self.slot), None)
            self._finish(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}


class VIEW3D_OT_fast_bone_shortcut_activate(Operator):
    bl_idname = "view3d.fast_bone_shortcut_activate"
    bl_label = "Activate Armature Shortcut"
    bl_options = {"REGISTER"}

    slot: bpy.props.IntProperty(min=1, max=5, default=1)

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return (
            context.mode == 'POSE' and wm.spine_mode_enabled
            and context.active_object is not None
            and context.active_object.type == 'ARMATURE'
        )

    def execute(self, context):
        arm = context.active_object
        targets = []
        for button in range(1, 4):
            name = _shortcut_target_name(arm, self.slot, button)
            if name and name not in targets and name in arm.pose.bones:
                targets.append(name)

        if not targets:
            return {'CANCELLED'}

        try:
            window_key = context.window.as_pointer()
        except Exception:
            window_key = 0
        state_key = (window_key, self.slot)
        idx = _shortcut_cycle_state.get(state_key, -1)
        idx = (idx + 1) % len(targets)
        _shortcut_cycle_state[state_key] = idx
        target_name = targets[idx]

        for pb in arm.pose.bones:
            try:
                pb.select = False
            except Exception:
                pass

        target = arm.pose.bones.get(target_name)
        if target is None:
            return {'CANCELLED'}
        try:
            target.select = True
        except Exception:
            return {'CANCELLED'}
        try:
            arm.data.bones.active = target.bone
        except Exception:
            pass
        _tag_redraw_view3d(context)
        return {'FINISHED'}

# ---------------------------------------------------------------------------
# N-Panel
# ---------------------------------------------------------------------------
class VIEW3D_PT_spine_bone_mode(Panel):
    bl_label = "Fast Bone Control"
    bl_idname = "VIEW3D_PT_spine_bone_mode"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Fast Bone"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        layout.operator(
            VIEW3D_OT_fast_bone_mode_toggle.bl_idname,
            text="Exit Fast Bone Control" if wm.spine_mode_enabled else "Enter Fast Bone Control",
            icon='CHECKBOX_HLT' if wm.spine_mode_enabled else 'CHECKBOX_DEHLT',
            depress=wm.spine_mode_enabled,
        )

        col = layout.column()
        col.enabled = wm.spine_mode_enabled

        col.separator()
        col.label(text="Right-Click Modes")
        row = col.row(align=True)
        row.prop(wm, "spine_mode_use_translate", text="Move", toggle=True)
        row.prop(wm, "spine_mode_use_rotate", text="Rotate", toggle=True)
        row.prop(wm, "spine_mode_use_scale", text="Scale", toggle=True)

        col.separator()
        snap_box = col.box()
        snap_box.label(text="Shift Snap Settings")
        snap_box.prop(wm, "spine_snap_move_angle_deg", text="Move Angle")
        snap_box.prop(wm, "spine_snap_rotate_angle_deg", text="Rotation Angle")
        snap_box.prop(wm, "spine_snap_scale_step", text="Scale Step")

        col.separator()
        shortcut_box = col.box()
        shortcut_box.label(text="Armature Shortcut")
        arm = context.active_object if context.active_object and context.active_object.type == 'ARMATURE' else None
        for slot in range(1, 6):
            row = shortcut_box.row(align=True)
            keycap = row.row(align=True)
            keycap.ui_units_x = 1.25
            keycap.alignment = 'CENTER'
            keycap.label(text=str(slot))
            for button in range(1, 4):
                prop_name = _shortcut_prop_name(slot, button)
                target_name = arm.data.get(prop_name, "") if arm else ""
                op = row.operator(
                    VIEW3D_OT_fast_bone_shortcut_pick_target.bl_idname,
                    text=target_name if target_name else "+",
                    depress=bool(target_name),
                )
                op.slot = slot
                op.button = button


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
classes = (
    POSE_OT_spine_cycle_mode,
    POSE_OT_spine_drag_transform,
    POSE_OT_spine_shift_snap_transform,
    VIEW3D_OT_fast_bone_mode_toggle,
    VIEW3D_OT_fast_bone_shortcut_pick_target,
    VIEW3D_OT_fast_bone_shortcut_activate,
    VIEW3D_PT_spine_bone_mode,
)

_draw_handler = None


def register():
    bpy.types.WindowManager.spine_bone_mode = EnumProperty(
        name="Fast Bone Transform Mode",
        items=[(m, MODE_LABELS[m], "") for m in MODES],
        default="TRANSLATE",
        update=_on_mode_update,
    )
    bpy.types.WindowManager.spine_mode_enabled = BoolProperty(
        name="Fast Bone Control",
        description="Enter the Fast Bone Control direct transform workflow",
        default=False,
        update=_on_enabled_update,
    )
    bpy.types.WindowManager.spine_mode_use_translate = BoolProperty(
        name="Include Move",
        description="Include Move in right-click mode cycling",
        default=True,
        update=_on_use_translate_update,
    )
    bpy.types.WindowManager.spine_mode_use_rotate = BoolProperty(
        name="Include Rotate",
        description="Include Rotate in right-click mode cycling",
        default=True,
        update=_on_use_rotate_update,
    )
    bpy.types.WindowManager.spine_mode_use_scale = BoolProperty(
        name="Include Scale",
        description="Include Scale in right-click mode cycling",
        default=False,
        update=_on_use_scale_update,
    )
    bpy.types.WindowManager.spine_snap_move_angle_deg = FloatProperty(
        name="Move Direction Angle",
        description="When Shift-moving, lock the view-plane movement direction to this angle increment",
        default=45.0,
        min=0.1,
        max=180.0,
        precision=1,
        step=10,
    )
    bpy.types.WindowManager.spine_snap_rotate_angle_deg = FloatProperty(
        name="Rotation Angle",
        description="When Shift-rotating, snap rotation to this angle increment",
        default=45.0,
        min=0.1,
        max=180.0,
        precision=1,
        step=10,
    )
    bpy.types.WindowManager.spine_snap_scale_step = FloatProperty(
        name="Scale Step",
        description="When Shift-scaling, snap scale around 1.0 using this increment",
        default=0.1,
        min=0.001,
        max=10.0,
        precision=3,
        step=1,
    )

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.utils.register_tool(SpineTweakTool, after={"builtin.select"}, separator=True)

    global _draw_handler
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_spine_mode_gizmo, (), "WINDOW", "POST_PIXEL"
    )


def unregister():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None

    bpy.utils.unregister_tool(SpineTweakTool)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.spine_bone_mode
    del bpy.types.WindowManager.spine_mode_enabled
    del bpy.types.WindowManager.spine_mode_use_translate
    del bpy.types.WindowManager.spine_mode_use_rotate
    del bpy.types.WindowManager.spine_mode_use_scale
    del bpy.types.WindowManager.spine_snap_move_angle_deg
    del bpy.types.WindowManager.spine_snap_rotate_angle_deg
    del bpy.types.WindowManager.spine_snap_scale_step


if __name__ == "__main__":
    register()
