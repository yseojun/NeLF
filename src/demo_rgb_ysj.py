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
import cv2,glob
import torchvision
from tqdm import tqdm
import imageio
import shutil
import statistics
import time

from sklearn.neighbors import NearestNeighbors

from Camera import Camera
from DataManager import DataManager

def rm_folder(path):
    if os.path.exists(path):
       files = glob.glob(path+'*')

       if(len(files)>0):
            for f in files:
                try:
                    shutil.rmtree(f)
                    
                except:
                    os.remove(f)
    #    os.makedirs(path)
    else:
        os.makedirs(path)

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')

class demo_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        data_root = args.data_dir
        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,depth_branch=False)
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.scale)) 

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'

        self.img_folder_test = 'demo_result_rgb/'+self.exp+'/'
        rm_folder(self.img_folder_test)

        self.load_check_points()

        self.model = self.model.cuda()

        image_paths = glob.glob(f"{data_img}/*"+args.img_form)
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0])
        self.w = int(sample_img.shape[1])

        self.uvst_whole   = np.load(f"{data_root}/uvsttrain.npy") 
        self.uvst_min = self.uvst_whole.min()
        self.uvst_max = self.uvst_whole.max()
        self.color_whole   = np.load(f"{data_root}/rgbtrain.npy")

        self.color_imgs = np.reshape(self.color_whole,(-1,self.h,self.w,3))

        self.fdepth = np.load(f"{data_root}/fdepthtrain.npy")

        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
       
        # UV 평면, ST 평면 거리 세팅
        self.uv_depth     = 1.0
        self.st_depth     = -1.0
       
        self.render_pose  = np.load(f"{data_root}/Render_posetrain.npy")
        self.intrinsic    = np.load(f"{data_root}/ktrain.npy")

        self.camera_pose = np.load(f"{data_root}/cam_posetrain.npy") 
        print(self.camera_pose.shape[0])

        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.camera_pose)
        self.data_root = data_root


    def gen_pose_llff(self):
        savename = f"{self.img_folder_test}/llff_pose"
        os.mkdir(savename)

        view_group       = []
        view_group_data  = []

        # 데이터셋 세팅 : (data_root, grid_size, image_size)
        self.datamanager = DataManager(base_path=self.data_root, grid_size=17, image_size=(256, 512))

        # 결과 카메라 세팅 : (x, y, z, theta, phi, fov, H, W)
        self.cam = Camera(x=0, y=0, z=-1, theta=0, phi=0, fov=90, H=256, W=512)

        
        model_times = []
        dataset_times = []

        with torch.no_grad():
            for i in tqdm(range(60), desc="Processing frames"):
                self.cam.set_c2w()
                self.cam.z += 2/60

                # 모델 코드
                model_start = time.time()
                data_uvst = self.cam.get_uvst()
                view_unit = self.render_sample_img(self.model,data_uvst,self.w,self.h,None,None,False)                
                view_unit *= 255           
                view_unit = view_unit.cpu().numpy().astype(np.uint8)

                view_unit = imageio.core.util.Array(view_unit)
                view_group.append(view_unit)
                model_end = time.time()
                model_time = model_end - model_start
                model_times.append(model_time)

                # 모델 - 이미지 저장
                cv2.imwrite(savename+"/"+str(i)+".png", cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
                tqdm.write(f'Frame {i} from model : {model_time:.4f} s')

                # --------------------------
                # 데이터셋 코드
                dataset_start = time.time()
                st = data_uvst[:, 2:4]
                output_fov = self.cam.get_output_fov()

                view_data_unit = self.datamanager.get_matched_rgb(st, output_fov)
                view_data_unit *= 255           
                view_data_unit = (view_data_unit * 255).astype(np.uint8)
                dataset_end = time.time()
                dataset_time = dataset_end - dataset_start
                dataset_times.append(dataset_time)
                tqdm.write(f'Frame {i} from dataset: {dataset_time:.4f} s')
                
                # 데이터셋 - 이미지 저장
                cv2.imwrite(savename+"/"+str(i)+"_data.png", cv2.cvtColor(view_data_unit,cv2.COLOR_RGB2BGR))

                view_unit = imageio.core.util.Array(view_unit)
                view_group_data.append(view_data_unit)

            avg_model_time = statistics.mean(model_times)
            avg_dataset_time = statistics.mean(dataset_times)
            print(f"모델 inference 평균 : {avg_model_time:.4f} s")
            print(f"데이터셋 평균 : {avg_dataset_time:.4f} s")

            # gif 저장
            imageio.mimsave(savename+".gif", view_group,fps=30)
            imageio.mimsave(savename+"_data.gif", view_group_data,fps=30)



    def render_sample_img(self,model,uvst, w, h, save_path=None,save_depth_path=None,save_flag=True):
         with torch.no_grad():
        
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

            pred_color = model(uvst)
  
            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
            
            return pred_color.reshape((h,w,3))

        
    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"./{self.checkpoints}/nelf-{self.iter}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
    
        self.model.load_state_dict(ckpt)
        
if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)
    unit.gen_pose_llff()

