import os
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)

# =============================================================================
# COORDINATE SYSTEM NOTES — READ BEFORE TOUCHING THIS FILE
# =============================================================================
#
# Internal convention (matches fbx_extract.py, bvh_extract.py, and viewer):
#   Quaternions stored as  (w, x, y, z)  — NOT scipy's default (x,y,z,w).
#   World space is Y-up, right-handed.
#
# FBX export targets:
#   force_z_up=False  →  Y-up  (Maya / Unity)      — curves written as-is
#   force_z_up=True   →  Z-up  (Blender / Unreal)  — root position and root
#                          orientation physically baked into Z-up space.
#
# WHY we bake instead of FbxAxisSystem.ConvertScene() or SetAxisSystem():
#   ConvertScene() rewrites baked curves and has documented bugs with
#   pre-rotations in the Python SDK.
#   SetAxisSystem() / SetInScene() write a metadata tag only — Blender
#   ignores that tag on import and always reads curve values as Z-up.
#   The conversion MUST be baked into the curve data itself.
#
# Z-up conversion (root joint ONLY):
#   Position:    (x,  y,  z)_yup  →  (x,  z, -y)_zup
#   Orientation: R_zup = Rx(+90°) * R_yup
#
# Non-root bone LOCAL rotations and bone-offset translations are NEVER
# modified — they live in joint-local space and are correct regardless
# of world up-axis.
#
# =============================================================================


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _wxyz_to_xyzw(q):
    """
    Reorder quaternion(s) from internal storage (w,x,y,z) to scipy (x,y,z,w).
    Accepts any shape (..., 4).  Returns same shape, new array.
    """
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _interpolate_track(track, global_frames):
    """
    Interpolate track data to exactly `global_frames` frames, honouring
    track.offset and track.scale.

    Returns
    -------
    pos   : (global_frames, J, 3)  world positions
    quats : (global_frames, J, 4)  world quats (w,x,y,z), or None
    """
    if track.positions is None:
        raise ValueError(f"Track '{track.name}' has no position data.")

    F, J, _ = track.positions.shape
    scale   = track.scale if track.scale != 0.0 else 1.0
    t       = np.clip((np.arange(global_frames) - track.offset) / scale, 0, F - 1)

    i0   = np.floor(t).astype(int)
    i1   = np.minimum(i0 + 1, F - 1)
    frac = (t - i0)[:, np.newaxis, np.newaxis]

    pos = track.positions[i0] * (1.0 - frac) + track.positions[i1] * frac

    if track.quaternions is None:
        return pos, None

    q0  = track.quaternions[i0]
    q1  = track.quaternions[i1]
    dot = np.sum(q0 * q1, axis=2, keepdims=True)
    q1  = np.where(dot < 0, -q1, q1)
    q   = q0 * (1.0 - frac) + q1 * frac
    q  /= np.maximum(np.linalg.norm(q, axis=2, keepdims=True), 1e-8)

    return pos, q


def _world_quats_to_local_euler(quats_wxyz, parent_indices, euler_order='xyz'):
    """
    Convert (F, J, 4) world quaternions (w,x,y,z) to (F, J, 3) local Euler
    angles in degrees using the skeleton hierarchy.

    Local rotation j = inv(world_rot[parent[j]]) * world_rot[j]
    Root joints (parent < 0) use their world rotation directly.
    """
    F, J, _ = quats_wxyz.shape
    xyzw       = _wxyz_to_xyzw(quats_wxyz.reshape(-1, 4))
    world_mats = R.from_quat(xyzw).as_matrix().reshape(F, J, 3, 3)

    local_mats = np.empty_like(world_mats)
    for j in range(J):
        p = parent_indices[j]
        if p < 0:
            local_mats[:, j] = world_mats[:, j]
        else:
            local_mats[:, j] = (
                np.transpose(world_mats[:, p], (0, 2, 1)) @ world_mats[:, j]
            )

    return (
        R.from_matrix(local_mats.reshape(-1, 3, 3))
         .as_euler(euler_order, degrees=True)
         .reshape(F, J, 3)
    )


# =============================================================================
# FBX EXPORT  —  PRIMARY / LOCKED DOWN
# =============================================================================

def export_timeline_to_fbx(session, filepath,
                            progress_callback=None,
                            include_mesh=False,
                            force_z_up=True):
    """
    Export all visible tracks to a single FBX file.

    Parameters
    ----------
    session           Session  — must expose .tracks (list) and .max_frame (int)
    filepath          str      — absolute output path including .fbx extension
    progress_callback callable — optional fn(percent:int) → bool; return True to cancel
    include_mesh      bool     — attach invisible proxy mesh so DCCs show the armature
    force_z_up        bool     — True: Z-up (Blender/Unreal), False: Y-up (Maya/Unity)

    Returns True on success, False if cancelled by progress_callback.
    Raises RuntimeError / ImportError / ValueError on hard failure.
    """

    # ------------------------------------------------------------------ guards
    try:
        import fbx
    except ImportError:
        raise ImportError(
            "Autodesk FBX Python SDK is not installed.\n"
            "Install fbxPIP or the official SDK wheel for Python 3.10."
        )

    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = int(session.max_frame)
    if global_frames <= 0:
        raise ValueError("Timeline is empty (max_frame == 0).")

    log.info(
        f"FBX export started | tracks={len(tracks)} | "
        f"frames={global_frames} | z_up={force_z_up} | path={filepath}"
    )

    # ------------------------------------------------- Z-up conversion helpers
    # Defined once per export call; only applied to root joint when force_z_up.
    zup_basis = R.from_euler('x', 90.0, degrees=True) if force_z_up else None

    def _pos_to_zup(p):
        """(x, y, z)_yup → (x, z, -y)_zup.  New array, input not mutated."""
        out = p.copy()
        out[..., 1] =  p[..., 2]
        out[..., 2] = -p[..., 1]
        return out

    # -------------------------------------------- FBX SDK manager / exporter
    manager = fbx.FbxManager.Create()
    fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)

    exporter_obj = fbx.FbxExporter.Create(manager, "")
    if not exporter_obj.Initialize(filepath, -1, manager.GetIOSettings()):
        err = exporter_obj.GetStatus().GetErrorString()
        exporter_obj.Destroy()
        manager.Destroy()
        raise RuntimeError(f"FBX exporter failed to initialise: {err}")

    scene = fbx.FbxScene.Create(manager, "MoCapScene")

    try:
        # --------------------------------------- scene-level global settings
        gs = scene.GetGlobalSettings()
        gs.SetTimeMode(fbx.FbxTime.EMode.eFrames60)

        # Timeline span — DCCs open with the correct playback range
        t_start = fbx.FbxTime(); t_start.SetFrame(0,            fbx.FbxTime.EMode.eFrames60)
        t_end   = fbx.FbxTime(); t_end.SetFrame(global_frames,  fbx.FbxTime.EMode.eFrames60)
        span    = fbx.FbxTimeSpan(); span.Set(t_start, t_end)
        gs.SetTimelineDefaultTimeSpan(span)

        # Axis metadata tag (Maya reads this; Blender ignores it — hence the bake)
        axis_system = fbx.FbxAxisSystem(
            fbx.FbxAxisSystem.EUpVector.eZAxis  if force_z_up
                else fbx.FbxAxisSystem.EUpVector.eYAxis,
            fbx.FbxAxisSystem.EFrontVector.eParityOdd,
            fbx.FbxAxisSystem.ECoordSystem.eRightHanded,
        )
        gs.SetAxisSystem(axis_system)

        # ----------------------------------------- animation stack / layer
        anim_stack = fbx.FbxAnimStack.Create(scene, "Take001")
        anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
        anim_stack.AddMember(anim_layer)
        anim_stack.SetLocalTimeSpan(span)

        fbx_time = fbx.FbxTime()

        # ================================================================
        # TRACK LOOP
        # ================================================================
        for t_idx, track in enumerate(tracks):

            # ---------------------------------- interpolate to timeline length
            pos, quats = _interpolate_track(track, global_frames)
            # pos   : (global_frames, J, 3)
            # quats : (global_frames, J, 4)  stored (w,x,y,z)

            parents = track.skeleton.parent_indices
            J       = len(parents)

            if quats is None:
                raise ValueError(
                    f"Track '{track.name}' has no rotation data — FBX export requires rotations."
                )

            # -------------------- precompute all local Euler angles (F, J, 3)
            # These are joint-LOCAL rotations. They are NEVER modified by the
            # Z-up conversion — that only affects the root position/orientation.
            local_euler = _world_quats_to_local_euler(quats, parents, euler_order='xyz')

            # -------------------- rest-pose (bind-pose) data
            rp_pos = (track.rest_pose_positions
                      if track.rest_pose_positions is not None
                      else track.positions[0])              # (J, 3)

            rq_raw = (track.rest_pose_quaternions
                      if track.rest_pose_quaternions is not None
                      else (track.quaternions[0]
                            if track.quaternions is not None else None))  # (J,4) or None

            if rq_raw is not None:
                rest_euler = _world_quats_to_local_euler(
                    rq_raw[np.newaxis], parents, euler_order='xyz'
                )[0]    # (J, 3)
            else:
                rest_euler = np.zeros((J, 3))

            # -------------------- root alignment constants
            track_rot = R.from_euler(
                'xyz', [track.rotate_x, track.rotate_y, track.rotate_z],
                degrees=True
            )
            aji    = track.align_joint_index
            f0_pos = track.positions[0, aji, :].copy()
            f0_pos[1] = 0.0     # strip XZ floor drift, preserve height

            # ================================================================
            # JOINT LOOP
            # ================================================================
            fbx_nodes = [None] * J

            for j in range(J):
                jname   = track.skeleton.joint_names[j]
                is_root = parents[j] < 0

                # Root named after track; children use joint_trackname
                node_name = track.name if is_root else f"{jname}_{track.name}"

                node      = fbx.FbxNode.Create(scene, node_name)
                skel_attr = fbx.FbxSkeleton.Create(scene, node_name)
                skel_attr.SetSkeletonType(
                    fbx.FbxSkeleton.EType.eRoot     if is_root
                    else fbx.FbxSkeleton.EType.eLimbNode
                )
                skel_attr.Size.Set(1.0)
                node.SetNodeAttribute(skel_attr)

                if is_root:
                    scene.GetRootNode().AddChild(node)
                else:
                    fbx_nodes[parents[j]].AddChild(node)
                fbx_nodes[j] = node

                # ---------------------------------- get animation curves
                curve_tx = node.LclTranslation.GetCurve(anim_layer, "X", True)
                curve_ty = node.LclTranslation.GetCurve(anim_layer, "Y", True)
                curve_tz = node.LclTranslation.GetCurve(anim_layer, "Z", True)
                curve_rx = node.LclRotation.GetCurve(anim_layer, "X", True)
                curve_ry = node.LclRotation.GetCurve(anim_layer, "Y", True)
                curve_rz = node.LclRotation.GetCurve(anim_layer, "Z", True)

                curve_tx.KeyModifyBegin(); curve_ty.KeyModifyBegin(); curve_tz.KeyModifyBegin()
                curve_rx.KeyModifyBegin(); curve_ry.KeyModifyBegin(); curve_rz.KeyModifyBegin()

                # Constant bone offset for non-root joints
                if not is_root:
                    bone_offset = track.positions[0, j] - track.positions[0, parents[j]]

                # ============================================================
                # FRAME LOOP
                # f == -1  →  rest / bind pose  (FBX time frame 0)
                # f == 0..global_frames-1  →  animation  (FBX time frame 1..N)
                # ============================================================
                for f in range(-1, global_frames):

                    if (f + 1) % 250 == 0:
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()

                    fbx_time.SetFrame(f + 1, fbx.FbxTime.EMode.eFrames60)

                    # Choose source data for this frame
                    if f == -1:
                        raw_pos_j = rp_pos[j].copy()
                        euler_j   = rest_euler[j]
                    else:
                        raw_pos_j = pos[f, j].copy()
                        euler_j   = local_euler[f, j]

                    # ------------------------------------------------ ROOT
                    if is_root:
                        # Step 1 — remove XZ floor-origin drift
                        rp = raw_pos_j - f0_pos

                        # Step 2 — apply UI rotation (rotate_x/y/z)
                        rp = track_rot.apply(rp)

                        # Step 3 — apply UI translation (translate_x/y/z)
                        rp[0] += track.translate_x
                        rp[1] += track.translate_y
                        rp[2] += track.translate_z

                        # Step 4 — Z-up: remap root POSITION
                        #   (x, y, z)_yup → (x, z, -y)_zup
                        if force_z_up:
                            rp = _pos_to_zup(rp)

                        k = curve_tx.KeyAdd(fbx_time)[0]; curve_tx.KeySet(k, fbx_time, float(rp[0]))
                        k = curve_ty.KeyAdd(fbx_time)[0]; curve_ty.KeySet(k, fbx_time, float(rp[1]))
                        k = curve_tz.KeyAdd(fbx_time)[0]; curve_tz.KeySet(k, fbx_time, float(rp[2]))

                        # Step 5 — build root orientation (UI rotation × local)
                        root_r = track_rot * R.from_euler('xyz', euler_j, degrees=True)

                        # Step 6 — Z-up: re-orient root by Rx(+90°)
                        #   Non-root joint rotations are NEVER touched here.
                        if force_z_up:
                            root_r = zup_basis * root_r

                        ex, ey, ez = root_r.as_euler('xyz', degrees=True)

                    # ------------------------------------------ NON-ROOT
                    else:
                        # Constant bone offset — same every frame
                        k = curve_tx.KeyAdd(fbx_time)[0]; curve_tx.KeySet(k, fbx_time, float(bone_offset[0]))
                        k = curve_ty.KeyAdd(fbx_time)[0]; curve_ty.KeySet(k, fbx_time, float(bone_offset[1]))
                        k = curve_tz.KeyAdd(fbx_time)[0]; curve_tz.KeySet(k, fbx_time, float(bone_offset[2]))

                        # Local rotation — untouched, no coordinate conversion
                        ex, ey, ez = euler_j

                    # ----------------------------------------- write rotation (all joints)
                    k = curve_rx.KeyAdd(fbx_time)[0]; curve_rx.KeySet(k, fbx_time, float(ex))
                    k = curve_ry.KeyAdd(fbx_time)[0]; curve_ry.KeySet(k, fbx_time, float(ey))
                    k = curve_rz.KeyAdd(fbx_time)[0]; curve_rz.KeySet(k, fbx_time, float(ez))

                # ------------------------------------------------ end frame loop
                curve_tx.KeyModifyEnd(); curve_ty.KeyModifyEnd(); curve_tz.KeyModifyEnd()
                curve_rx.KeyModifyEnd(); curve_ry.KeyModifyEnd(); curve_rz.KeyModifyEnd()

            # ================================================================ end joint loop

            # ---------------------------------------------------- bind pose
            t0 = fbx.FbxTime(); t0.SetFrame(0, fbx.FbxTime.EMode.eFrames60)
            bind_pose = fbx.FbxPose.Create(scene, f"BindPose_{track.name}")
            bind_pose.SetIsBindPose(True)
            for node in fbx_nodes:
                bind_pose.Add(node, fbx.FbxMatrix(node.EvaluateGlobalTransform(t0)))
            scene.AddPose(bind_pose)

            # ----------------------------------------- optional proxy mesh
            if include_mesh:
                mname = f"ProxyMesh_{track.name}"
                mesh  = fbx.FbxMesh.Create(scene, mname)
                mesh.InitControlPoints(3)
                mesh.SetControlPointAt(fbx.FbxVector4(0.0,   0.0,   0.0, 0.0), 0)
                mesh.SetControlPointAt(fbx.FbxVector4(0.001, 0.0,   0.0, 0.0), 1)
                mesh.SetControlPointAt(fbx.FbxVector4(0.0,   0.001, 0.0, 0.0), 2)
                mesh.BeginPolygon()
                mesh.AddPolygon(0); mesh.AddPolygon(1); mesh.AddPolygon(2)
                mesh.EndPolygon()

                mesh_node = fbx.FbxNode.Create(scene, f"{mname}_Node")
                mesh_node.SetNodeAttribute(mesh)
                scene.GetRootNode().AddChild(mesh_node)

                cluster = fbx.FbxCluster.Create(scene, f"Cluster_{track.name}")
                cluster.SetLink(fbx_nodes[0])
                cluster.SetLinkMode(fbx.FbxCluster.eTotalOne)
                cluster.AddControlPointIndex(0, 1.0)
                cluster.AddControlPointIndex(1, 1.0)
                cluster.AddControlPointIndex(2, 1.0)
                cluster.SetTransformMatrix(mesh_node.EvaluateGlobalTransform(t0))
                cluster.SetTransformLinkMatrix(fbx_nodes[0].EvaluateGlobalTransform(t0))

                skin = fbx.FbxSkin.Create(scene, f"Skin_{track.name}")
                skin.AddCluster(cluster)
                mesh.AddDeformer(skin)
                bind_pose.Add(mesh_node, fbx.FbxMatrix(mesh_node.EvaluateGlobalTransform(t0)))

            # ------------------------------------------------- progress
            if progress_callback:
                pct = int(((t_idx + 1) / len(tracks)) * 100)
                if progress_callback(pct):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    log.warning("FBX export cancelled. Partial file deleted.")
                    return False

        # ================================================================ end track loop

        exporter_obj.Export(scene)
        log.info("FBX export complete.")
        return True

    finally:
        exporter_obj.Destroy()
        manager.Destroy()


# =============================================================================
# BVH EXPORT  —  SECONDARY (preserved)
# =============================================================================

def export_timeline_to_bvh(session, filepath, progress_callback=None):
    """Export all visible tracks to a single BVH file (multiple ROOT blocks)."""

    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = session.max_frame
    if global_frames <= 0:
        raise ValueError("Timeline is empty.")

    ref_fps       = tracks[0].fps
    export_frames = global_frames + 1   # frame -1 = rest pose

    log.info(f"BVH export: {len(tracks)} track(s), {export_frames} frames → {filepath}")

    interp_pos, interp_quat = [], []
    for track in tracks:
        p, q = _interpolate_track(track, global_frames)
        interp_pos.append(p)
        interp_quat.append(q)

    cancelled = False
    try:
        with open(filepath, "w") as f:
            f.write("HIERARCHY\n")
            for t_idx, track in enumerate(tracks):
                _bvh_write_hierarchy(f, track, t_idx)

            f.write("MOTION\n")
            f.write(f"Frames: {export_frames}\n")
            f.write(f"Frame Time: {1.0 / ref_fps:.6f}\n")

            for frame in range(-1, global_frames):
                parts = []
                for t_idx, track in enumerate(tracks):
                    if frame == -1:
                        rp = (track.rest_pose_positions
                              if track.rest_pose_positions is not None
                              else track.positions[0])
                        rq = (track.rest_pose_quaternions
                              if track.rest_pose_quaternions is not None
                              else (track.quaternions[0] if track.quaternions is not None else None))
                        if rq is not None and rq.ndim == 3:
                            rq = rq[0]
                        parts.append(_bvh_frame_channels(track, rp, rq))
                    else:
                        q = interp_quat[t_idx][frame] if interp_quat[t_idx] is not None else None
                        parts.append(_bvh_frame_channels(track, interp_pos[t_idx][frame], q))

                f.write(" ".join(parts) + "\n")

                if progress_callback and frame % 10 == 0:
                    pct = max(0, min(100, int(((frame + 2) / export_frames) * 100)))
                    if progress_callback(pct):
                        cancelled = True
                        break
    finally:
        if cancelled:
            try:
                os.remove(filepath)
            except OSError:
                pass
            log.warning("BVH export cancelled. Partial file deleted.")
            return False

    log.info("BVH export complete.")
    return True


def _bvh_write_hierarchy(f, track, t_idx):
    pos0    = track.positions[0]
    parents = track.skeleton.parent_indices

    def write_node(j, depth):
        name     = track.skeleton.joint_names[j]
        indent   = "  " * depth
        children = [c for c, p in enumerate(parents) if p == j]
        is_root  = parents[j] < 0

        if is_root:
            f.write(f"{indent}ROOT {name}_T{t_idx}\n")
        elif not children:
            f.write(f"{indent}End Site\n")
        else:
            f.write(f"{indent}JOINT {name}_T{t_idx}\n")

        f.write(f"{indent}{{\n")
        offset = [0.0, 0.0, 0.0] if is_root else list(pos0[j] - pos0[parents[j]])
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

        if is_root:
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        elif children:
            f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")

        for child in children:
            write_node(child, depth + 1)
        f.write(f"{indent}}}\n")

    for r in [i for i, p in enumerate(parents) if p < 0]:
        write_node(r, 0)


def _bvh_frame_channels(track, pos, quats):
    if quats is None:
        return ""

    parents = track.skeleton.parent_indices
    J       = len(parents)

    xyzw       = _wxyz_to_xyzw(quats)
    world_mats = R.from_quat(xyzw).as_matrix()

    local_mats = np.zeros((J, 3, 3))
    for i in range(J):
        p = parents[i]
        local_mats[i] = world_mats[i] if p < 0 else world_mats[p].T @ world_mats[i]

    local_euler = R.from_matrix(local_mats).as_euler('zxy', degrees=True)

    channels  = []
    track_rot = R.from_euler('xyz', [track.rotate_x, track.rotate_y, track.rotate_z], degrees=True)
    aji       = track.align_joint_index
    f0        = track.positions[0, aji, :].copy()
    f0[1]     = 0.0

    for r in [i for i, p in enumerate(parents) if p < 0]:
        rp = pos[r].copy() - f0
        rp = track_rot.apply(rp)
        rp[0] += track.translate_x
        rp[1] += track.translate_y
        rp[2] += track.translate_z
        channels.extend([f"{rp[0]:.4f}", f"{rp[1]:.4f}", f"{rp[2]:.4f}"])

    for i in range(J):
        if parents[i] < 0 or any(p == i for p in parents):
            channels.extend([f"{local_euler[i,0]:.4f}",
                              f"{local_euler[i,1]:.4f}",
                              f"{local_euler[i,2]:.4f}"])

    return " ".join(channels)