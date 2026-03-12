import os
import logging
import math
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

log = logging.getLogger(__name__)

def export_timeline_to_bvh(session, filepath, progress_callback=None):
    """
    Exports the visible timeline tracks into a single BVH file with multiple ROOT nodes.
    Applies trim, sub-frame offset (interpolation), scale, and 3D translation/rotation.
    """
    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = session.max_frame
    if global_frames <= 0:
        raise ValueError("Timeline is empty.")

    log.info(f"Exporting BVH with {len(tracks)} tracks and {global_frames} frames to {filepath}")

    with open(filepath, "w") as f:
        f.write("HIERARCHY\n")
        
        # Structure to hold interpolation results so we only compute once
        interpolated_pos = []
        interpolated_quat = []

        # 1. HIERARCHY
        for t_idx, track in enumerate(tracks):
            _write_bvh_hierarchy(f, track, t_idx)
            
            # Interpolate per track
            pos, quats = _interpolate_track(track, global_frames)
            interpolated_pos.append(pos)
            interpolated_quat.append(quats)

        # 2. MOTION
        export_frames = global_frames + 1
        f.write("MOTION\n")
        f.write(f"Frames: {export_frames}\n")
        f.write(f"Frame Time: {1.0 / track.fps:.6f}\n")

        for frame in range(-1, global_frames):
            line_parts = []
            for t_idx, track in enumerate(tracks):
                if frame == -1:
                    r_pos = track.rest_pose_positions if track.rest_pose_positions is not None else track.positions[0]
                    r_quat = track.rest_pose_quaternions if track.rest_pose_quaternions is not None else (track.quaternions[0] if track.quaternions is not None else None)
                    line_parts.append(_compute_bvh_frame_channels(track, r_pos, r_quat))
                else:
                    line_parts.append(_compute_bvh_frame_channels(track, interpolated_pos[t_idx][frame], interpolated_quat[t_idx][frame]))
            
            f.write(" ".join(line_parts) + "\n")
            
            if progress_callback and frame % 10 == 0:
                percent = int(((frame + 1) / export_frames) * 100)
                if progress_callback(percent):
                    f.close()
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    log.warning("BVH Export cancelled by user. Partial file deleted.")
                    return False

    log.info("BVH Export complete.")
    return True

def _write_bvh_hierarchy(f, track, t_idx):
    def write_node(j_idx, depth):
        name = track.skeleton.joint_names[j_idx]
        indent = "  " * depth
        
        children = [i for i, p in enumerate(track.skeleton.parent_indices) if p == j_idx]
        is_root = track.skeleton.parent_indices[j_idx] < 0
        
        if is_root:
            f.write(f"{indent}ROOT {name}_T{t_idx}\n")
        elif not children:
            f.write(f"{indent}End Site\n")
        else:
            f.write(f"{indent}JOINT {name}_T{t_idx}\n")
            
        f.write(f"{indent}{{\n")
        
        # Offsets
        pos0 = track.positions[0]
        if is_root:
            offset = [0.0, 0.0, 0.0]
        else:
            parent_idx = track.skeleton.parent_indices[j_idx]
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

def _interpolate_track(track, global_frames):
    F, J, _ = track.positions.shape
    local_times = (np.arange(global_frames) - track.offset) / track.scale
    local_times = np.clip(local_times, 0, F - 1)
    
    idx0 = np.floor(local_times).astype(int)
    idx1 = np.minimum(idx0 + 1, F - 1)
    frac = (local_times - idx0)[:, np.newaxis, np.newaxis]
    
    # Position lerp
    pos = track.positions[idx0] * (1.0 - frac) + track.positions[idx1] * frac
    
    if track.quaternions is None:
        return pos, None
        
    # Slerp is expensive to loop, so we vectorize using scipy Rotation or simple nlerp
    # Simple Nlerp for quaternion array of shape (F, J, 4)
    q0 = track.quaternions[idx0]
    q1 = track.quaternions[idx1]
    
    # Ensure shortest path
    dotProduct = np.sum(q0 * q1, axis=2, keepdims=True)
    q1 = np.where(dotProduct < 0, -q1, q1)
    
    q_interp = q0 * (1.0 - frac) + q1 * frac
    q_norm = np.linalg.norm(q_interp, axis=2, keepdims=True)
    q_interp = q_interp / np.maximum(q_norm, 1e-8)
    
    return pos, q_interp

def _compute_bvh_frame_channels(track, pos, quats):
    if quats is None:
        return ""
    
    parents = track.skeleton.parent_indices
    J = len(parents)
    
    # We must formulate LOCAL euler angles.
    # quats contains WORLD quaternions.
    # local_q = parent_world_q_inv * child_world_q
    world_r = R.from_quat(quats)
    world_mats = world_r.as_matrix() # (J, 3, 3)
    
    local_mats = np.zeros((J, 3, 3))
    for i in range(J):
        p = parents[i]
        if p < 0:
            local_mats[i] = world_mats[i]
        else:
            # Inv(parent) * child
            local_mats[i] = world_mats[p].T @ world_mats[i]
            
    # BVH standard is local eulers in ZXY order.
    # Scipy euler uses intrinsic or extrinsic. For ZXY it's 'zxy' or 'ZXY'. 
    local_euler = R.from_matrix(local_mats).as_euler('zxy', degrees=True)
    
    channels = []
    
    # Root position
    # The BVH root needs to incorporate translation, align offsets, and rotations from Track.
    roots = [i for i, p in enumerate(parents) if p < 0]
    for r in roots:
        root_pos = pos[r]
        
        # Apply alignment offset (from track.aligned_positions logic)
        aji = track.align_joint_index
        frame0_pos = track.positions[0, aji, :].copy()
        frame0_pos[1] = 0.0
        root_pos = root_pos - frame0_pos
        
        # Apply Rotation
        track_rot = R.from_euler('xyz', [track.rotate_x, track.rotate_y, track.rotate_z], degrees=True)
        root_pos = track_rot.apply(root_pos)
        
        # Apply Translation
        root_pos[0] += track.translate_x
        root_pos[1] += track.translate_y
        root_pos[2] += track.translate_z
        
        channels.extend([f"{root_pos[0]:.4f}", f"{root_pos[1]:.4f}", f"{root_pos[2]:.4f}"])
        
    for i in range(J):
        children = [c for c, p in enumerate(parents) if p == i]
        if parents[i] < 0 or children:  # Has channels
            channels.extend([f"{local_euler[i, 0]:.4f}", f"{local_euler[i, 1]:.4f}", f"{local_euler[i, 2]:.4f}"])
            
    return " ".join(channels)


def export_timeline_to_fbx(session, filepath, progress_callback=None):
    """
    Exports the visible timeline tracks into an FBX file with multiple root nodes.
    Applies trim, sub-frame offset (interpolation), scale, and 3D translation/rotation.
    """
    try:
        import fbx
    except ImportError:
        raise ImportError("Autodesk FBX SDK is not installed.")

    tracks = [t for t in session.tracks if t is not None and t.visible]
    if not tracks:
        raise ValueError("No visible tracks to export.")

    global_frames = int(session.max_frame)
    if global_frames <= 0:
        raise ValueError("Timeline is empty.")

    log.info(f"Exporting FBX with {len(tracks)} tracks and {global_frames} frames to {filepath}")

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(filepath, -1, manager.GetIOSettings()):
        err = exporter.GetStatus().GetErrorString()
        manager.Destroy()
        raise RuntimeError(f"FBX Exporter failed: {err}")

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

            # Get local matrices for all frames
            world_r = R.from_quat(quats.reshape(-1, 4))
            world_mats = world_r.as_matrix().reshape(global_frames, J, 3, 3)  # (F, J, 3, 3)

            local_mats = np.zeros((global_frames, J, 3, 3))
            for i in range(J):
                p = parents[i]
                if p < 0:
                    local_mats[:, i] = world_mats[:, i]
                else:
                    parent_inv = np.transpose(world_mats[:, p], axes=(0, 2, 1))
                    local_mats[:, i] = parent_inv @ world_mats[:, i]

            local_euler = R.from_matrix(local_mats.reshape(-1, 3, 3)).as_euler('xyz', degrees=True).reshape(global_frames, J, 3)

            fbx_nodes = []
            for i in range(J):
                name = f"{track.skeleton.joint_names[i]}_T{t_idx}"
                node = fbx.FbxNode.Create(scene, name)
                skeleton = fbx.FbxSkeleton.Create(scene, "skeleton")
                if parents[i] < 0:
                    skeleton.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)
                    scene.GetRootNode().AddChild(node)
                else:
                    skeleton.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
                    fbx_nodes[parents[i]].AddChild(node)
                    
                skeleton.Size.Set(1.0)
                node.SetNodeAttribute(skeleton)
                fbx_nodes.append(node)

                curve_t_x = node.LclTranslation.GetCurve(anim_layer, "X", True)
                curve_t_y = node.LclTranslation.GetCurve(anim_layer, "Y", True)
                curve_t_z = node.LclTranslation.GetCurve(anim_layer, "Z", True)

                curve_r_x = node.LclRotation.GetCurve(anim_layer, "X", True)
                curve_r_y = node.LclRotation.GetCurve(anim_layer, "Y", True)
                curve_r_z = node.LclRotation.GetCurve(anim_layer, "Z", True)

                curve_t_x.KeyModifyBegin()
                curve_t_y.KeyModifyBegin()
                curve_t_z.KeyModifyBegin()
                curve_r_x.KeyModifyBegin()
                curve_r_y.KeyModifyBegin()
                curve_r_z.KeyModifyBegin()

                is_root = parents[i] < 0
                if is_root:
                    # Precompute track rotations for all frames
                    track_rot = R.from_euler('xyz', [track.rotate_x, track.rotate_y, track.rotate_z], degrees=True)
                
                    aji = track.align_joint_index
                    f0_pos = track.positions[0, aji, :].copy()
                    f0_pos[1] = 0.0
                
                else:
                    offset_val = track.positions[0, i] - track.positions[0, parents[i]]
                
                for f in range(-1, global_frames):
                    if f % 250 == 0:
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()
                    
                    fbx_time.SetFrame(f + 1, fbx.FbxTime.EMode.eFrames60)

                    if f == -1:
                        rp = track.rest_pose_positions if track.rest_pose_positions is not None else track.positions[0]
                        rq = track.rest_pose_quaternions if track.rest_pose_quaternions is not None else (track.quaternions[0] if track.quaternions is not None else None)
                    
                        if is_root:
                            rt = rp[i].copy()
                        else:
                            rt = None
                        
                        if rq is not None:
                            w_r = R.from_quat(rq)
                            w_m = w_r.as_matrix()
                            p = parents[i]
                            if p < 0:
                                l_m = w_m[i]
                            else:
                                l_m = w_m[p].T @ w_m[i]
                            r_eul = R.from_matrix(l_m).as_euler('xyz', degrees=True)
                        else:
                            r_eul = [0, 0, 0]

                        p_f = rt
                        e_f = r_eul
                    else:
                        p_f = pos[f, i].copy() if is_root else None
                        e_f = local_euler[f, i]

                    if is_root:
                        p_f -= f0_pos
                        p_f = track_rot.apply(p_f)
                    
                        p_f[0] += track.translate_x
                        p_f[1] += track.translate_y
                        p_f[2] += track.translate_z

                        kx = curve_t_x.KeyAdd(fbx_time)[0]
                        curve_t_x.KeySet(kx, fbx_time, float(p_f[0]))
                        ky = curve_t_y.KeyAdd(fbx_time)[0]
                        curve_t_y.KeySet(ky, fbx_time, float(p_f[1]))
                        kz = curve_t_z.KeyAdd(fbx_time)[0]
                        curve_t_z.KeySet(kz, fbx_time, float(p_f[2]))

                        rr = R.from_euler('xyz', e_f, degrees=True)
                        final_e = (track_rot * rr).as_euler('xyz', degrees=True)
                        e_x, e_y, e_z = final_e
                    else:
                        kx = curve_t_x.KeyAdd(fbx_time)[0]
                        curve_t_x.KeySet(kx, fbx_time, float(offset_val[0]))
                        ky = curve_t_y.KeyAdd(fbx_time)[0]
                        curve_t_y.KeySet(ky, fbx_time, float(offset_val[1]))
                        kz = curve_t_z.KeyAdd(fbx_time)[0]
                        curve_t_z.KeySet(kz, fbx_time, float(offset_val[2]))

                        e_x, e_y, e_z = e_f

                    kx = curve_r_x.KeyAdd(fbx_time)[0]
                    curve_r_x.KeySet(kx, fbx_time, float(e_x))
                    ky = curve_r_y.KeyAdd(fbx_time)[0]
                    curve_r_y.KeySet(ky, fbx_time, float(e_y))
                    kz = curve_r_z.KeyAdd(fbx_time)[0]
                    curve_r_z.KeySet(kz, fbx_time, float(e_z))

                curve_t_x.KeyModifyEnd()
                curve_t_y.KeyModifyEnd()
                curve_t_z.KeyModifyEnd()
                curve_r_x.KeyModifyEnd()
                curve_r_y.KeyModifyEnd()
                curve_r_z.KeyModifyEnd()

                if progress_callback:
                    percent = int(((t_idx + 1) / len(tracks)) * 100)
                    if progress_callback(percent):
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                        log.warning("FBX Export cancelled by user. Partial file deleted.")
                        return False

        exporter.Export(scene)
        log.info("FBX Export complete.")
        return True
        
    finally:
        exporter.Destroy()
        manager.Destroy()
