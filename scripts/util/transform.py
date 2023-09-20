import pyquaternion
import numpy as np


nusc2opecv = np.array([
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]])

def rotation_translation_to_pose_np(rotation, translation):
    
    pose = np.eye(4)

    pose[:3, :3] = pyquaternion.Quaternion(rotation).rotation_matrix

    pose[:3, 3] = translation

    return pose