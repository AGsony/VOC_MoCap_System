import os
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quat_wxyz_to_xyzw(q):
    """
    Convert quaternion array from (w,x,y,z) storage convention to scipy's (x,y,z,w).
    Works on any shape (..., 4).
    """
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _interpolate_track(track, global_frames):
    """
    Return (pos, quats) arrays interpolated to global_frames length,
    applying track.offset and track.scale.

    BUG FIX: track.scale=0 previously caused ZeroDivisionError.
    BUG FIX: track.positions being None no longer raises AttributeError.
    """
    if track.positions is None:
        raise ValueError(f"Track '{track.name}' has no position data.")

    F, J, _ = track.positions.shape
    scale = track.scale if track.scale != 0.0 else 1.0   # guard divide-by-zero
    local_times = (np.arange(global_frames) - track.offset) / scale
    local_times = np.clip(local_times, 0, F - 1)

    idx0 = np.floor(local_times).astype(int)
    idx1 = np.minimum(idx0 + 1, F - 1)
    frac = (local_times - idx0)[:, np.newaxis, np.newaxis]

    # Position lerp
    pos = track.positions[idx0] * (1.0 - frac) + track.positions[idx1] * frac

    if track.quaternions is None:
        return pos, None

    q0 = track.quaternions[idx0]   # (global_frames, J, 4)  stored w,x,y,z
    q1 = track.quaternions[idx1]

    # Ensure shortest-path interpolation
    dot = np.sum(q0 * q1, axis=2, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)

    # Normalized linear interpolation (nlerp) — fast and stable enough for 60 fps
    q_interp = q0 * (1.0 - frac) + q1 * frac
    q_norm = np.linalg.norm(q_interp, axis=2, keepdims=True)
    q_interp = q_interp / np.maximum(q_norm, 1e-8)

    return pos, q_interp


def _world_quats_to_local_euler(quats_wxyz, parent_indices, euler_order='xyz'):
    """
    Convert per-joint world quaternions (w,x,y,z) to local Euler angles (degrees).

    BUG FIX: was passing (w,x,y,z) directly to scipy which expects (x,y,z,w),
    silently corrupting every rotation.

    Returns ndarray shape (..., J, 3).
    """
    leading = quats_wxyz.shape[:-2]   # e.g. (F,) or ()
    J = quats_wxyz.shape[-2]

    flat = quats_wxyz.reshape(-1, J, 4)
    N = flat.shape[0]

    xyzw = _quat_wxyz_to_xyzw(flat.reshape(-1, 4))
    world_mats = R.from_quat(xyzw).as_matrix().reshape(N, J, 3, 3)

    local_mats = np.zeros_like(world_mats)
    for i in range(J):
        p = parent_indices[i]
        if p < 0:
            local_mats[:, i] = world_mats[:, i]
        else:
            parent_inv = np.transpose(world_mats[:, p], axes=(0, 2, 1))
            local_mats[:, i] = parent_inv @ world_mats[:, i]

    euler = R.from_matrix(local_mats.reshape(-1, 3, 3)) \
              .as_euler(euler_order, degrees=True) \
              .reshape(N, J, 3)

    return euler.reshape(*leading, J, 3)


# ---------------------------------------------------------------------------
# BVH Export
# ---------------------------------------------------------------------------

def export_timeline_to_bvh(session, filepath, progress_callback=None):
    """
    Export all visible timeline tracks to a single BVH file.

    BUG FIX: 'track' variable leaked from inner loop into Frame Time line —
    would reference the LAST track's fps instead of a consistent value,
    and would crash if tracks list was empty (already guarded above but
    the leak was still a latent bug).

    BUG FIX: file handle was manually closed inside the loop on cancellation
    while also being managed by the 'with' block — double-close.

    BUG FIX: progress percent calculation used (frame+1)/export_frames but
    frame starts at -1, so first tick reported negative percent.
    """
    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = session.max_frame
    if global_frames <= 0:
        raise ValueError("Timeline is empty.")

    # Use reference track fps for frame time — consistent, not leaked from loop
    ref_fps = tracks[0].fps
    export_frames = global_frames + 1   # +1 for rest-pose frame at index 0

    log.info(f"Exporting BVH: {len(tracks)} track(s), {export_frames} frames → {filepath}")

    # Pre-interpolate all tracks before opening the file
    interpolated_pos = []
    interpolated_quat = []
    for track in tracks:
        pos, quats = _interpolate_track(track, global_frames)
        interpolated_pos.append(pos)
        interpolated_quat.append(quats)

    cancelled = False
    try:
        with open(filepath, "w") as f:
            f.write("HIERARCHY\n")
            for t_idx, track in enumerate(tracks):
                _write_bvh_hierarchy(f, track, t_idx)

            f.write("MOTION\n")
            f.write(f"Frames: {export_frames}\n")
            f.write(f"Frame Time: {1.0 / ref_fps:.6f}\n")

            for frame in range(-1, global_frames):
                line_parts = []
                for t_idx, track in enumerate(tracks):
                    if frame == -1:
                        r_pos  = track.rest_pose_positions  if track.rest_pose_positions  is not None else track.positions[0]
                        r_quat = track.rest_pose_quaternions if track.rest_pose_quaternions is not None \
                                 else (track.quaternions[0:1] if track.quaternions is not None else None)
                        # r_quat shape must be (J,4) for the helper — squeeze if needed
                        if r_quat is not None and r_quat.ndim == 3:
                            r_quat = r_quat[0]
                        line_parts.append(_compute_bvh_frame_channels(track, r_pos, r_quat))
                    else:
                        line_parts.append(_compute_bvh_frame_channels(
                            track,
                            interpolated_pos[t_idx][frame],
                            interpolated_quat[t_idx][frame] if interpolated_quat[t_idx] is not None else None,
                        ))

                f.write(" ".join(line_parts) + "\n")

                if progress_callback and frame % 10 == 0:
                    # frame runs -1..global_frames-1; shift so percent is always 0-100
                    percent = int(((frame + 2) / export_frames) * 100)
                    percent = max(0, min(100, percent))
                    if progress_callback(percent):
                        cancelled = True
                        break

    finally:
        if cancelled:
            try:
                os.remove(filepath)
            except OSError:
                pass
            log.warning("BVH export cancelled by user. Partial file deleted.")
            return False

    log.info("BVH export complete.")
    return True


def _write_bvh_hierarchy(f, track, t_idx):
    """Write HIERARCHY section for one track."""
    pos0 = track.positions[0]

    def write_node(j_idx, depth):
        name    = track.skeleton.joint_names[j_idx]
        indent  = "  " * depth
        parents = track.skeleton.parent_indices
        children = [i for i, p in enumerate(parents) if p == j_idx]
        is_root  = parents[j_idx] < 0

        if is_root:
            f.write(f"{indent}ROOT {name}_T{t_idx}\n")
        elif not children:
            f.write(f"{indent}End Site\n")
        else:
            f.write(f"{indent}JOINT {name}_T{t_idx}\n")

        f.write(f"{indent}{{\n")

        if is_root:
            offset = [0.0, 0.0, 0.0]
        else:
            parent_idx = parents[j_idx]
            offset = pos0[j_idx] - pos0[parent_idx]

        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

        if is_root:
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        elif children:
            f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")

        for child in children:
            write_node(child, depth + 1)

        f.write(f"{indent}}}\n")

    roots = [i for i, p in enumerate(track.skeleton.parent_indices) if p < 0]
    for r in roots:
        write_node(r, 0)


def _compute_bvh_frame_channels(track, pos, quats):
    """
    Compute the BVH channel values for one frame of one track.

    BUG FIX: quats passed as (w,x,y,z) directly to R.from_quat which
    expects (x,y,z,w) — corrupted all rotations silently.

    BUG FIX: root_pos was mutated in-place (root_pos -= frame0_pos then
    root_pos[0] += ...) while pos[r] is a numpy view — could corrupt the
    source array. Now uses .copy() explicitly.
    """
    if quats is None:
        return ""

    parents = track.skeleton.parent_indices
    J = len(parents)

    # BUG FIX: reorder (w,x,y,z) → (x,y,z,w) before passing to scipy
    quats_xyzw = _quat_wxyz_to_xyzw(quats)           # (J, 4)
    world_mats = R.from_quat(quats_xyzw).as_matrix()  # (J, 3, 3)

    local_mats = np.zeros((J, 3, 3))
    for i in range(J):
        p = parents[i]
        if p < 0:
            local_mats[i] = world_mats[i]
        else:
            local_mats[i] = world_mats[p].T @ world_mats[i]

    # BVH standard rotation order: ZXY
    local_euler = R.from_matrix(local_mats).as_euler('zxy', degrees=True)  # (J, 3)

    channels = []

    track_rot = R.from_euler('xyz', [track.rotate_x, track.rotate_y, track.rotate_z], degrees=True)
    aji = track.align_joint_index
    frame0_xz = track.positions[0, aji, :].copy()
    frame0_xz[1] = 0.0

    roots = [i for i, p in enumerate(parents) if p < 0]
    for r in roots:
        root_pos = pos[r].copy()           # BUG FIX: explicit copy to avoid mutating source array
        root_pos -= frame0_xz
        root_pos  = track_rot.apply(root_pos)
        root_pos[0] += track.translate_x
        root_pos[1] += track.translate_y
        root_pos[2] += track.translate_z
        channels.extend([f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}"])

    for i in range(J):
        children = [c for c, p in enumerate(parents) if p == i]
        if parents[i] < 0 or children:
            channels.extend([
                f"{local_euler[i, 0]:.4f}",
                f"{local_euler[i, 1]:.4f}",
                f"{local_euler[i, 2]:.4f}",
            ])

    return " ".join(channels)


# ---------------------------------------------------------------------------
# FBX Export
# ---------------------------------------------------------------------------

def export_timeline_to_fbx(session, filepath, progress_callback=None,
                            include_mesh=False, force_z_up=True):
    """
    Export all visible timeline tracks to a single FBX file.

    BUG FIX: quaternion (w,x,y,z) → (x,y,z,w) reorder before all scipy calls.
    BUG FIX: ConvertScene() physically mutated baked curves → replaced with
             SetInScene() which writes axis metadata only.
    BUG FIX: manager.Destroy() was called in finally but exporter.Destroy()
             was also in finally — if exporter.Initialize failed, exporter
             was destroyed twice (once in error path, once in finally).
             Now exporter is only destroyed in finally.
    BUG FIX: curve_r_z.KeySet was never called — the kz KeyAdd line for
             rotation Z was present but the KeySet call was missing, leaving
             the Z rotation curve empty for every joint on every frame.
    BUG FIX: axis system block was inside the per-track loop — it ran once
             per track, converting the scene multiple times. Moved outside.
    BUG FIX: KeyModifyEnd for curve_r_z was missing from the finally cleanup
             (only X and Y had it), leaving the curve in an inconsistent state.
    """
    try:
        import fbx
    except ImportError:
        raise ImportError("Autodesk FBX Python SDK is not installed.")

    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = int(session.max_frame)
    if global_frames <= 0:
        raise ValueError("Timeline is empty.")

    log.info(f"Exporting FBX: {len(tracks)} track(s), {global_frames} frames → {filepath}")

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    exporter_obj = fbx.FbxExporter.Create(manager, "")
    if not exporter_obj.Initialize(filepath, -1, manager.GetIOSettings()):
        err = exporter_obj.GetStatus().GetErrorString()
        exporter_obj.Destroy()
        manager.Destroy()
        raise RuntimeError(f"FBX Exporter failed to initialise: {err}")

    scene = fbx.FbxScene.Create(manager, "Scene")

    try:
        scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime.EMode.eFrames60)

        anim_stack = fbx.FbxAnimStack.Create(scene, "TimelineStack")
        anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
        anim_stack.AddMember(anim_layer)

        fbx_time = fbx.FbxTime()

        for t_idx, track in enumerate(tracks):
            pos, quats = _interpolate_track(track, global_frames)

            parents = track.skeleton.parent_indices
            J = len(parents)

            # --- Precompute local Euler angles for all frames ---
            # BUG FIX: was passing (w,x,y,z) to scipy which expects (x,y,z,w)
            quats_xyzw = _quat_wxyz_to_xyzw(quats.reshape(-1, 4))
            world_mats = R.from_quat(quats_xyzw).as_matrix().reshape(global_frames, J, 3, 3)

            local_mats = np.zeros((global_frames, J, 3, 3))
            for i in range(J):
                p = parents[i]
                if p < 0:
                    local_mats[:, i] = world_mats[:, i]
                else:
                    parent_inv = np.transpose(world_mats[:, p], axes=(0, 2, 1))
                    local_mats[:, i] = parent_inv @ world_mats[:, i]

            local_euler = (
                R.from_matrix(local_mats.reshape(-1, 3, 3))
                 .as_euler('xyz', degrees=True)
                 .reshape(global_frames, J, 3)
            )

            fbx_nodes = [None] * J

            for i in range(J):
                joint_name  = track.skeleton.joint_names[i]
                name_prefix = f"{joint_name}_T{t_idx}"
                node        = fbx.FbxNode.Create(scene, name_prefix)
                skeleton_attr = fbx.FbxSkeleton.Create(scene, joint_name)

                fbx_nodes[i] = node   # store before parenting

                if parents[i] < 0:
                    skeleton_attr.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)
                    scene.GetRootNode().AddChild(node)
                else:
                    skeleton_attr.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
                    fbx_nodes[parents[i]].AddChild(node)

                skeleton_attr.Size.Set(1.0)
                node.SetNodeAttribute(skeleton_attr)

                curve_t_x = node.LclTranslation.GetCurve(anim_layer, "X", True)
                curve_t_y = node.LclTranslation.GetCurve(anim_layer, "Y", True)
                curve_t_z = node.LclTranslation.GetCurve(anim_layer, "Z", True)
                curve_r_x = node.LclRotation.GetCurve(anim_layer, "X", True)
                curve_r_y = node.LclRotation.GetCurve(anim_layer, "Y", True)
                curve_r_z = node.LclRotation.GetCurve(anim_layer, "Z", True)

                curve_t_x.KeyModifyBegin(); curve_t_y.KeyModifyBegin(); curve_t_z.KeyModifyBegin()
                curve_r_x.KeyModifyBegin(); curve_r_y.KeyModifyBegin(); curve_r_z.KeyModifyBegin()

                is_root = parents[i] < 0

                if is_root:
                    track_rot = R.from_euler('xyz',
                                             [track.rotate_x, track.rotate_y, track.rotate_z],
                                             degrees=True)
                    aji    = track.align_joint_index
                    f0_pos = track.positions[0, aji, :].copy()
                    f0_pos[1] = 0.0
                else:
                    offset_val = track.positions[0, i] - track.positions[0, parents[i]]

                for f in range(-1, global_frames):
                    if f % 250 == 0:
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()

                    fbx_time.SetFrame(f + 1, fbx.FbxTime.EMode.eFrames60)

                    # --- Determine position and euler for this frame ---
                    if f == -1:
                        # Rest-pose frame
                        rp = (track.rest_pose_positions
                              if track.rest_pose_positions is not None
                              else track.positions[0])
                        rq = (track.rest_pose_quaternions
                              if track.rest_pose_quaternions is not None
                              else (track.quaternions[0] if track.quaternions is not None else None))

                        p_f = rp[i].copy() if is_root else None

                        if rq is not None:
                            # BUG FIX: reorder (w,x,y,z) → (x,y,z,w)
                            rq_1d = rq[i] if rq.ndim == 2 else rq   # handle (J,4) or (4,)
                            rq_xyzw = np.array([rq_1d[1], rq_1d[2], rq_1d[3], rq_1d[0]])
                            # Compute local rotation
                            par = parents[i]
                            if par < 0:
                                l_mat = R.from_quat(rq_xyzw).as_matrix()
                            else:
                                if rq.ndim == 2:
                                    rq_par_xyzw = np.array([rq[par,1], rq[par,2], rq[par,3], rq[par,0]])
                                else:
                                    rq_par_xyzw = rq_xyzw  # fallback
                                w_mat_i   = R.from_quat(rq_xyzw).as_matrix()
                                w_mat_par = R.from_quat(rq_par_xyzw).as_matrix()
                                l_mat = w_mat_par.T @ w_mat_i
                            e_f = R.from_matrix(l_mat).as_euler('xyz', degrees=True)
                        else:
                            e_f = np.zeros(3)
                    else:
                        p_f = pos[f, i].copy() if is_root else None
                        e_f = local_euler[f, i]

                    # --- Write translation keys ---
                    if is_root:
                        p_f -= f0_pos
                        p_f  = track_rot.apply(p_f)
                        p_f[0] += track.translate_x
                        p_f[1] += track.translate_y
                        p_f[2] += track.translate_z

                        kx = curve_t_x.KeyAdd(fbx_time)[0]; curve_t_x.KeySet(kx, fbx_time, float(p_f[0]))
                        ky = curve_t_y.KeyAdd(fbx_time)[0]; curve_t_y.KeySet(ky, fbx_time, float(p_f[1]))
                        kz = curve_t_z.KeyAdd(fbx_time)[0]; curve_t_z.KeySet(kz, fbx_time, float(p_f[2]))

                        rr      = R.from_euler('xyz', e_f, degrees=True)
                        final_e = (track_rot * rr).as_euler('xyz', degrees=True)
                        e_x, e_y, e_z = final_e
                    else:
                        kx = curve_t_x.KeyAdd(fbx_time)[0]; curve_t_x.KeySet(kx, fbx_time, float(offset_val[0]))
                        ky = curve_t_y.KeyAdd(fbx_time)[0]; curve_t_y.KeySet(ky, fbx_time, float(offset_val[1]))
                        kz = curve_t_z.KeyAdd(fbx_time)[0]; curve_t_z.KeySet(kz, fbx_time, float(offset_val[2]))
                        e_x, e_y, e_z = e_f

                    # --- Write rotation keys ---
                    kx = curve_r_x.KeyAdd(fbx_time)[0]; curve_r_x.KeySet(kx, fbx_time, float(e_x))
                    ky = curve_r_y.KeyAdd(fbx_time)[0]; curve_r_y.KeySet(ky, fbx_time, float(e_y))
                    # BUG FIX: curve_r_z.KeySet was NEVER called in original — Z rotation was blank
                    kz = curve_r_z.KeyAdd(fbx_time)[0]; curve_r_z.KeySet(kz, fbx_time, float(e_z))

                curve_t_x.KeyModifyEnd(); curve_t_y.KeyModifyEnd(); curve_t_z.KeyModifyEnd()
                curve_r_x.KeyModifyEnd(); curve_r_y.KeyModifyEnd(); curve_r_z.KeyModifyEnd()

            # --- Bind pose ---
            bind_pose = fbx.FbxPose.Create(scene, f"BindPose_T{t_idx}")
            bind_pose.SetIsBindPose(True)
            for n in fbx_nodes:
                bind_pose.Add(n, fbx.FbxMatrix(n.EvaluateGlobalTransform(fbx.FbxTime(0))))
            scene.AddPose(bind_pose)

            # --- Optional proxy mesh (Option B) ---
            if include_mesh and len(fbx_nodes) > 0:
                mesh_name = f"ProxyMesh_T{t_idx}"
                mesh = fbx.FbxMesh.Create(scene, mesh_name)

                mesh.InitControlPoints(3)
                mesh.SetControlPointAt(fbx.FbxVector4(0,     0,     0, 0), 0)
                mesh.SetControlPointAt(fbx.FbxVector4(0.001, 0,     0, 0), 1)
                mesh.SetControlPointAt(fbx.FbxVector4(0,     0.001, 0, 0), 2)

                mesh.BeginPolygon(); mesh.AddPolygon(0); mesh.AddPolygon(1); mesh.AddPolygon(2); mesh.EndPolygon()

                mesh_node = fbx.FbxNode.Create(scene, f"{mesh_name}_Node")
                mesh_node.SetNodeAttribute(mesh)
                scene.GetRootNode().AddChild(mesh_node)

                skin    = fbx.FbxSkin.Create(scene, f"Skin_T{t_idx}")
                cluster = fbx.FbxCluster.Create(scene, f"Cluster_T{t_idx}")
                cluster.SetLink(fbx_nodes[0])
                cluster.SetLinkMode(fbx.FbxCluster.eTotalOne)
                cluster.AddControlPointIndex(0, 1.0)
                cluster.AddControlPointIndex(1, 1.0)
                cluster.AddControlPointIndex(2, 1.0)
                cluster.SetTransformMatrix(mesh_node.EvaluateGlobalTransform(fbx.FbxTime(0)))
                cluster.SetTransformLinkMatrix(fbx_nodes[0].EvaluateGlobalTransform(fbx.FbxTime(0)))
                skin.AddCluster(cluster)
                mesh.AddDeformer(skin)

                bind_pose.Add(mesh_node, fbx.FbxMatrix(mesh_node.EvaluateGlobalTransform(fbx.FbxTime(0))))

            if progress_callback:
                percent = int(((t_idx + 1) / len(tracks)) * 100)
                if progress_callback(percent):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    log.warning("FBX export cancelled by user. Partial file deleted.")
                    return False

        # BUG FIX: axis system block was INSIDE the per-track loop — it ran once
        # per track, calling ConvertScene multiple times and compounding the rotation.
        # Moved here, outside the loop, and replaced ConvertScene with SetInScene
        # which writes metadata only and does NOT mutate any transform curves.
        if force_z_up:
            axis_system = fbx.FbxAxisSystem(
                fbx.FbxAxisSystem.EUpVector.eZAxis,
                fbx.FbxAxisSystem.EFrontVector.eParityOdd,
                fbx.FbxAxisSystem.ECoordSystem.eRightHanded,
            )
        else:
            axis_system = fbx.FbxAxisSystem(
                fbx.FbxAxisSystem.EUpVector.eYAxis,
                fbx.FbxAxisSystem.EFrontVector.eParityOdd,
                fbx.FbxAxisSystem.ECoordSystem.eRightHanded,
            )
        axis_system.SetInScene(scene)

        exporter_obj.Export(scene)
        log.info("FBX export complete.")
        return True

    finally:
        exporter_obj.Destroy()
        manager.Destroy()
