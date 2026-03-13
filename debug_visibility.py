import sys
import os
import numpy as np

# Add the local directory to sys.path so we can import the module correctly
sys.path.insert(0, r"c:\Users\bxdpo\Desktop\VOC_MoCap_System")

from PySide6.QtWidgets import QApplication
from mocap_studio.gui.main_window import MainWindow

def test_visibility():
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Create fake track data
    from mocap_studio.core.track import Track
    from mocap_studio.core.skeleton import Skeleton
    
    track = Track()
    track.name = "TestTrack"
    track.fps = 60.0
    track.frame_count = 10
    
    skel = Skeleton()
    skel.joint_names = ["Root", "Spine"]
    skel.parent_indices = [-1, 0]
    track.skeleton = skel
    
    track.positions = np.zeros((10, 2, 3))
    track.quaternions = np.zeros((10, 2, 4))
    track.quaternions[:,:,0] = 1.0 # default w
    
    window._session.load_track(0, track)
    track.auto_setup()
    
    # Try updating UI & Viewer
    try:
        window._update_track_ui(0)
        window._update_viewer()
    except Exception as e:
        print(f"ERROR calling update: {e}")
        return
        
    viewer_td = window._viewer._tracks_data[0]
    if viewer_td is None:
        print("Viewer track data is NONE!")
    else:
        print(f"Visible flag: {viewer_td['visible']}")
        print(f"Positions shape: {viewer_td['positions'].shape}")
        
    app.quit()

if __name__ == "__main__":
    test_visibility()
