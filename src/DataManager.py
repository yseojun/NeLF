import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

class DataManager:
    def __init__(self, base_path='/data/hmjung/data_backup/NeuLF_rgb/dataset/stanford_half/beans/', grid_size=17, image_size=(256, 512)):
        self.base_path = base_path

        self.image_size = image_size
        self.grid_size = grid_size
        
        self.uvst_data, self.rgb_data = self.load_data()
        self.knn_st = NearestNeighbors(n_neighbors=1)
        self.preprocess()

        self.knn_fov = NearestNeighbors(n_neighbors=1)
        self.calculate_data_fov()

    def load_data(self):
        print("데이터 로드 중...")
        uvst_train = np.load(os.path.join(self.base_path, 'uvsttrain.npy'))
        uvst_val = np.load(os.path.join(self.base_path, 'uvstval.npy'))
        rgb_train = np.load(os.path.join(self.base_path, 'rgbtrain.npy'))
        rgb_val = np.load(os.path.join(self.base_path, 'rgbval.npy'))

        uvst_data = np.concatenate((uvst_train, uvst_val), axis=0)
        rgb_data = np.concatenate((rgb_train, rgb_val), axis=0)
        print("데이터 로드 완료.")
        return uvst_data, rgb_data

    def preprocess(self):
        print("데이터 전처리 중...")
        num_images = self.grid_size * self.grid_size
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        tmp_uvst = np.zeros_like(self.uvst_data)
        tmp_rgb = np.zeros_like(self.rgb_data)
        st_values = np.zeros((num_images, 2))
        
        for i in range(num_images):
            st_values[i] = self.uvst_data[i * pixels_per_image, 2:4]
        
        sorted_indices = np.lexsort((st_values[:, 1], st_values[:, 0]))
        
        for new_idx, old_idx in enumerate(sorted_indices):
            start = old_idx * pixels_per_image
            end = (old_idx + 1) * pixels_per_image
            new_start = new_idx * pixels_per_image
            new_end = (new_idx + 1) * pixels_per_image
            
            tmp_uvst[new_start:new_end] = self.uvst_data[start:end]
            tmp_rgb[new_start:new_end] = self.rgb_data[start:end]
            
            st_values[new_idx] = self.uvst_data[start, 2:4]
        
        self.uvst_data = tmp_uvst
        self.rgb_data = tmp_rgb
        self.knn_st.fit(st_values)
        
        print("데이터 전처리 완료.")

    def calculate_data_fov(self):
        print("데이터 FOV 배열 세팅 중...")
        H, W = self.image_size
        fov = 90

        aspect_ratio = W / H
        fov_rad = np.radians(fov)
        
        if aspect_ratio >= 1:
            phi = np.linspace(-np.tan(fov_rad/2)/aspect_ratio, np.tan(fov_rad/2)/aspect_ratio, H)
            theta = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), W)
        else:
            phi = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), H)
            theta = np.linspace(-np.tan(fov_rad/2)*aspect_ratio, np.tan(fov_rad/2)*aspect_ratio, W)
        
        theta, phi = np.meshgrid(theta, phi)
        fov_array = np.stack([theta, phi], axis=-1)
        fov_array = fov_array.reshape(-1, 2)

        self.knn_fov.fit(fov_array)
        return fov_array
        
    def find_nearest_st_arr(self, st_arr):
        distances, indices = self.knn_st.kneighbors(st_arr)
        return indices
    
    def match_output_to_data_fov(self, output_fov):
        # print("Output FOV를 Data FOV와 매칭 중...")
        output_fov_flat = output_fov.reshape(-1, 2)
        distances, indices = self.knn_fov.kneighbors(output_fov_flat)
        # print("FOV 매칭 완료.")
        return indices

    def get_matched_rgb(self, st_arr, output_fov):
        H, W = output_fov.shape[:2]
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        # print("nearest st 찾는 중...")
        image_indices = self.find_nearest_st_arr(st_arr)
        
        # print("fov 매칭 중...")
        pixel_coords = self.match_output_to_data_fov(output_fov)
        
        # print("index 계산 중...")
        indices = image_indices * pixels_per_image + pixel_coords
        
        # print("rgb 데이터 추출 중...")
        matched_rgb = self.rgb_data[indices]
        matched_rgb = 1 - matched_rgb
        matched_rgb = matched_rgb.reshape(H, W, 3)
        
        # print("RGB 데이터 추출 완료.")
        return matched_rgb