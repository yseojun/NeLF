import numpy as np
from sklearn.neighbors import NearestNeighbors

class Camera:
    def __init__(self, x=0, y=0, z=0, theta=0, phi=0, fov=90, H=256, W=512):
        self.H = H
        self.W = W
        self.K = self.calculate_intrinsic()

        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.phi = phi

        self.fov = fov

        self.c2w = self.set_c2w()

        self.uv_depth = 2.0
        self.st_depth = 0
        
        self.plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]), (H * W, 3))
        self.plane_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),(self.H * self.W, 3))
        self.plane_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),(self.H * self.W, 3))
                
    def calculate_intrinsic(self):
        cx = self.W / 2
        cy = self.H / 2

        focal_length = np.sqrt(self.W**2 + self.H**2)

        intrinsic = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        return intrinsic
    
    def get_uvst(self):
        aspect = self.W / self.H
        uv_scale = 1.0  
        st_scale = 0.1

        u = np.linspace(-1, 1, self.W, dtype='float32')
        v = np.linspace(1, -1, self.H, dtype='float32') / aspect
        vu = np.meshgrid(u, v)

        u = vu[0] * uv_scale
        v = vu[1] * uv_scale
        
        cam_trans = self.get_camera_translation()[0]  # [x, y, z]
        
        s = np.ones_like(vu[0]) * cam_trans[0] * st_scale
        t = np.ones_like(vu[1]) * cam_trans[1] * st_scale
        
        uvst = np.stack((u, v, s, t), axis=-1)
        uvst = np.reshape(uvst, (-1, 4))
        
        return uvst
    
    def get_camera_translation(self):
        current_T = self.c2w[:3,-1].T
        current_T = np.expand_dims(current_T,axis=0)
        return current_T
    
    def set_c2w(self):
        theta = np.radians(self.theta)
        phi = np.radians(self.phi)
        psi = np.pi
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        
        T = np.array([self.x, self.y, self.z])
        
        c2w = np.zeros((3, 4))
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        self.c2w = c2w
        return c2w