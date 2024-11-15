# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from src.model import Nerf4D_relu_ps
from utils import rm_folder, rm_folder_keep
import cv2,glob
import torchvision
import math
from src.cam_view import rayPlaneInter
from tqdm import tqdm
from utils import eval_uvst
from src.utils import get_rays_np
import imageio
from sklearn.neighbors import NearestNeighbors
import time
import onnxruntime

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')

rendering_times = []

class demo_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        # data_root
        data_root = args.data_dir
        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,depth_branch=False)
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.scale)) 

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = '/data/ysj/neulf_result/'+self.exp+'/checkpoints/'

        self.img_folder_test = 'demo_result_rgb/'+self.exp+'/'
        rm_folder(self.img_folder_test)

        self.load_check_points()

        self.model = self.model.cuda()

        # height and width
        image_paths = glob.glob(f"{data_img}/*"+args.img_form)
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0])
        self.w = int(sample_img.shape[1])

        # get max 
        self.uvst_whole   = np.load(f"{data_root}/uvsttrain.npy") 
        self.uvst_min = self.uvst_whole.min()
        self.uvst_max = self.uvst_whole.max()
        self.color_whole   = np.load(f"{data_root}/rgbtrain.npy")

        self.color_imgs = np.reshape(self.color_whole,(-1,self.h,self.w,3))
        # input val
        

        self.fdepth = np.load(f"{data_root}/fdepthtrain.npy") # center object

        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
        self.min_u,self.max_u,self.min_v,self.max_v,self.min_s,self.max_s,self.min_t,self.max_t = eval_uvst(rays_whole)
       
        self.uv_depth     = 0.0
        self.st_depth     = -self.fdepth
       
        # render pose
        self.render_pose  = np.load(f"{data_root}/Render_posetrain.npy")#render path spiral
        self.intrinsic    = np.load(f"{data_root}/ktrain.npy")

        # load camera pose
        self.camera_pose = np.load(f"{data_root}/cam_posetrain.npy") 
        print(self.camera_pose.shape[0])

        # find nearest view in training
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.camera_pose)

    def gen_pose_llff(self):
        savename = f"{self.img_folder_test}/llff_pose"

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))

        view_group = []

        num_frame = self.render_pose.shape[0]

        with torch.no_grad():
            for i, c2w in enumerate(tqdm(self.render_pose)):
                start_time = time.time()
                ray_o, ray_d = get_rays_np(self.h, self.w, self.intrinsic, c2w)

                ray_o = np.reshape(ray_o,(-1,3))
                ray_d = np.reshape(ray_d,(-1,3))

                plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

                p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
                p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

                inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
                inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)

                data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)

                data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max - self.uvst_min) * 2 -1.0
                
                view_unit = self.render_sample_img(self.model,data_uvst,self.w,self.h,None,None,False)
                
                view_unit *= 255
           
                view_unit = view_unit.cpu().numpy().astype(np.uint8)
                end_time = time.time()
                rendering_time = end_time - start_time
                rendering_times.append(rendering_time)

                out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
                
                view_unit = imageio.core.util.Array(view_unit)
                
                view_group.append(view_unit)

            imageio.mimsave(savename+".gif", view_group,fps=30)
            avg_rendering_time = np.mean(rendering_times)
            print(f"평균 렌더링 시간: {avg_rendering_time * 1000:.2f} 밀리초")
            return avg_rendering_time

    def gen_pose_llff_onnx(self):
        savename = f"{self.img_folder_test}/llff_pose_onnx"

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))

        view_group = []

        num_frame = self.render_pose.shape[0]
        rendering_times_onnx = []

        for i, c2w in enumerate(tqdm(self.render_pose)):
            start_time = time.time()
            ray_o, ray_d = get_rays_np(self.h, self.w, self.intrinsic, c2w)

            ray_o = np.reshape(ray_o,(-1,3))
            ray_d = np.reshape(ray_d,(-1,3))

            plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

            p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
            p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

            inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
            inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)

            data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)

            data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max - self.uvst_min) * 2 -1.0
            
            view_unit = self.render_sample_img_onnx(data_uvst, self.w, self.h)

            view_unit *= 255
       
            view_unit = view_unit.astype(np.uint8)
            
            end_time = time.time()
            rendering_time = end_time - start_time
            rendering_times_onnx.append(rendering_time)

            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
            view_unit = imageio.core.util.Array(view_unit)
            
            view_group.append(view_unit)

        imageio.mimsave(savename+".gif", view_group,fps=30)
        
        avg_rendering_time_onnx = np.mean(rendering_times_onnx)
        print(f"ONNX 모델 평균 렌더링 시간: {avg_rendering_time_onnx * 1000:.2f} 밀리초")
        
        return avg_rendering_time_onnx

    def render_sample_img(self,model,uvst, w, h, save_path=None,save_depth_path=None,save_flag=True):
         with torch.no_grad():
        
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

            pred_color = model(uvst)
  
            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
            
            return pred_color.reshape((h,w,3)) #,pred_depth_norm.reshape((h,w,1))

        
    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"{self.checkpoints}/nelf-{self.iter}.pth"
        # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
    
        self.model.load_state_dict(ckpt)
        
    def load_onnx_model(self):
        self.onnx_model = onnxruntime.InferenceSession(f"{self.exp}.onnx")

    def render_sample_img_onnx(self, uvst, w, h):
        uvst = uvst.astype(np.float32)
        ort_inputs = {self.onnx_model.get_inputs()[0].name: uvst}
        ort_outs = self.onnx_model.run(None, ort_inputs)
        pred_color = ort_outs[0]
        return pred_color.reshape((h,w,3))

    def compare_inference_time(self):
        self.load_onnx_model()
        
        print("PyTorch 모델로 렌더링 중...")
        avg_time_pytorch = self.gen_pose_llff()
        
        print("ONNX 모델로 렌더링 중...")
        avg_time_onnx = self.gen_pose_llff_onnx()
        
        print(f"\nPyTorch 모델 평균 렌더링 시간: {avg_time_pytorch * 1000:.2f} 밀리초")
        print(f"ONNX 모델 평균 렌더링 시간: {avg_time_onnx * 1000:.2f} 밀리초")
        print(f"속도 향상: {(avg_time_pytorch / avg_time_onnx - 1) * 100:.2f}%")

if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)
    unit.compare_inference_time()
 