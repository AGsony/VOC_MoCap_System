import os
import sys
import numpy as np
import logging

# Add mocap_studio to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from mocap_studio.core.session import Session
from mocap_studio.core.track import Track
from mocap_studio.core.skeleton import Skeleton
from mocap_studio.core.exporter import export_timeline_to_fbx

logging.basicConfig(level=logging.INFO)

def run_test():
    session = Session()
    
    # Create dummy skeleton:
    # Root -> Spine -> Head
    skel = Skeleton(
        joint_names=["Root", "Spine", "Head"],
        parent_indices=[-1, 0, 1]
    )
    
    # Dummy data: 10 frames
    F = 10
    J = 3
    positions = np.zeros((F, J, 3))
    
    # Move root along Z over time
    positions[:, 0, 2] = np.linspace(0, 100, F)
    # Spine is 10 units up
    positions[:, 1, 1] = 10.0
    positions[:, 1, 2] = np.linspace(0, 100, F)
    # Head is 20 units up
    positions[:, 2, 1] = 20.0
    positions[:, 2, 2] = np.linspace(0, 100, F)
    
    quaternions = np.zeros((F, J, 4))
    quaternions[:, :, 3] = 1.0 # w=1 identity quat
    
    track = Track(
        name="TestTrack",
        source_path="dummy.fbx",
        fps=60,
        frame_count=F,
        skeleton=skel,
        positions=positions,
        quaternions=quaternions
    )
    track.align_joint = "Root"
    track.visible = True
    
    session.tracks[0] = track
    
    out_path = os.path.abspath("test_output.fbx")
    print(f"Exporting to {out_path} ...")
    
    export_timeline_to_fbx(session, out_path)
    print(f"Success! Built {out_path}")

if __name__ == "__main__":
    run_test()
