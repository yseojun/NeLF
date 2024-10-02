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

def compare_and_save_diff(img1, img2, frame_num, savename, name1, name2):
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    if np.any(diff > 0):
        tqdm.write(f'프레임 {frame_num}: {name1}와 {name2} 사이에 색상 차이 발견')
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        tqdm.write(f'  최대 차이: {max_diff}, 평균 차이: {avg_diff:.2f}')
        
        max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)
        tqdm.write(f'  최대 차이 위치: {max_diff_pos}')
        tqdm.write(f'  {name1} 값: {img1[max_diff_pos]}')
        tqdm.write(f'  {name2} 값: {img2[max_diff_pos]}')
        
        diff_image = np.clip(diff * 10, 0, 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
        cv2.imwrite(f"{savename}/{frame_num}_{name1}_vs_{name2}_diff.png", diff_colored)
        return diff_colored
    return 

class demo_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        data_root = args.data_dir
        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,depth_branch=False)
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.scale)) 

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = '/data/ysj/neulf_result/'+self.exp+'/checkpoints/'

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

        view_group = []
        view_group_data = []
        view_group_data_4 = []

        self.datamanager = DataManager(base_path=self.data_root, grid_size=17, image_size=(256, 512))
        self.cam = Camera(x=0, y=0, z=0, theta=-10, phi=0, fov=90, H=256, W=512)

        model_times = []
        dataset_times = []
        dataset_times_4 = []

        with torch.no_grad():
            for i in tqdm(range(60), desc="프레임 처리 중"):
                self.cam.set_c2w()
                self.cam.theta += 20/60

                # 모델 코드
                model_time, view_unit = self.run_model_save_img(i, savename)
                model_times.append(model_time)
                view_group.append(view_unit)

                # 데이터셋 코드 (1-NN)
                dataset_time, view_data_unit = self.run_data_save_img(i, savename)
                dataset_times.append(dataset_time)
                view_group_data.append(view_data_unit)

                # 데이터셋 코드 (4-NN)
                dataset_time_4, view_data_unit_4 = self.run_data_save_img_point_4(i, savename)
                dataset_times_4.append(dataset_time_4)
                view_group_data_4.append(view_data_unit_4)

            # 평균 시간 계산 및 출력
            self.print_average_times(model_times, dataset_times, dataset_times_4)

            # gif 저장
            self.save_gifs(savename, view_group, view_group_data, view_group_data_4)

    # 모델 inference
    def run_model_save_img(self, frame_num, savename):
        start_time = time.time()
        data_uvst = self.cam.get_uvst()
        view_unit = self.render_sample_img(self.model, data_uvst, self.w, self.h, None, None, False)                
        view_unit = (view_unit * 255).cpu().numpy().astype(np.uint8)
        view_unit = imageio.core.util.Array(view_unit)
        
        cv2.imwrite(f"{savename}/{frame_num}.png", cv2.cvtColor(view_unit, cv2.COLOR_RGB2BGR))
        
        end_time = time.time()
        model_time = end_time - start_time
        tqdm.write(f'Frame {frame_num} from model: {model_time:.4f} s')
        
        return model_time, view_unit

    # 데이터셋 1 point
    def run_data_save_img(self, frame_num, savename):
        start_time = time.time()
        data_uvst = self.cam.get_uvst()
        st = data_uvst[:, 2:4]
        output_fov = self.cam.get_output_fov()

        view_data_unit = self.datamanager.get_matched_rgb_1(st, output_fov)
        view_data_unit = (view_data_unit * 255).astype(np.uint8)
        
        cv2.imwrite(f"{savename}/{frame_num}_data.png", cv2.cvtColor(view_data_unit, cv2.COLOR_RGB2BGR))
        
        end_time = time.time()
        dataset_time = end_time - start_time
        tqdm.write(f'Frame {frame_num} from dataset: {dataset_time:.4f} s')
        
        return dataset_time, imageio.core.util.Array(view_data_unit)

    # 데이터셋 4 point
    def run_data_save_img_point_4(self, frame_num, savename):
        start_time = time.time()
        data_uvst = self.cam.get_uvst()
        st = data_uvst[:, 2:4]
        output_fov = self.cam.get_output_fov()

        view_data_unit_4 = self.datamanager.get_matched_rgb_4(st, output_fov)
        view_data_unit_4 = (view_data_unit_4 * 255).astype(np.uint8)
        
        cv2.imwrite(f"{savename}/{frame_num}_data_4.png", cv2.cvtColor(view_data_unit_4, cv2.COLOR_RGB2BGR))
        
        end_time = time.time()
        dataset_time_4 = end_time - start_time
        tqdm.write(f'Frame {frame_num} from dataset (4-NN): {dataset_time_4:.4f} s')
        
        return dataset_time_4, imageio.core.util.Array(view_data_unit_4)

    def print_average_times(self, model_times, dataset_times, dataset_times_4):
        avg_model_time = statistics.mean(model_times)
        avg_dataset_time = statistics.mean(dataset_times)
        avg_dataset_time_4 = statistics.mean(dataset_times_4)
        print()
        print(f"모델 inference 평균 : {avg_model_time:.4f} s")
        print(f"데이터셋 평균 : {avg_dataset_time:.4f} s")
        print(f"데이터셋 4포인트 가중치 평균 : {avg_dataset_time_4:.4f} s")

    def save_gifs(self, savename, view_group, view_group_data, view_group_data_4):
        imageio.mimsave(f"{savename}.gif", view_group, fps=30)
        imageio.mimsave(f"{savename}_data.gif", view_group_data, fps=30)
        imageio.mimsave(f"{savename}_data_4.gif", view_group_data_4, fps=30)

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
            ckpt_name = f"{self.checkpoints}/nelf-{self.iter}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
    
        self.model.load_state_dict(ckpt)
        
if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)
    unit.gen_pose_llff()