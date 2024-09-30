import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

class DataManager:
    def __init__(self, base_path='/data/hmjung/data_backup/NeuLF_rgb/dataset/stanford_half/beans/', grid_size=17, image_size=(256, 512)):
        self.base_path = base_path

        self.image_size = image_size
        self.grid_size = grid_size
        
        self.uvst_data, self.rgb_data = self.load_data()
        
        # 기존 1개의 최근접 이웃용 KNN
        self.knn_st_1 = NearestNeighbors(n_neighbors=1)
        self.knn_fov_1 = NearestNeighbors(n_neighbors=1)
        
        # 새로운 4개의 최근접 이웃용 KNN
        self.knn_st_4 = NearestNeighbors(n_neighbors=4)
        
        self.preprocess()
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
        
        # 1개의 최근접 이웃용 KNN 학습
        self.knn_st_1.fit(st_values)
        
        # 4개의 최근접 이웃용 KNN 학습
        self.knn_st_4.fit(st_values)
        
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

        # 1개의 최근접 이웃용 KNN 학습
        self.knn_fov_1.fit(fov_array)
        
        return fov_array
            
    def find_nearest_st_arr_1(self, st_arr):
        distances, indices = self.knn_st_1.kneighbors(st_arr)
        return indices
    
    def find_nearest_st_arr_4(self, st_arr):
        distances, indices = self.knn_st_4.kneighbors(st_arr)
        return distances, indices

    def match_output_to_data_fov_1(self, output_fov):
        # print("Output FOV를 Data FOV와 매칭 중 (1 NN)...")
        output_fov_flat = output_fov.reshape(-1, 2)
        distances, indices = self.knn_fov_1.kneighbors(output_fov_flat)
        # print("FOV 매칭 완료 (1 NN).")
        return indices

    def get_matched_rgb_1(self, st_arr, output_fov):
        H, W = output_fov.shape[:2]
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        # 1개의 최근접 이웃 찾기
        image_indices = self.find_nearest_st_arr_1(st_arr)
        pixel_coords = self.match_output_to_data_fov_1(output_fov)
        
        # 인덱스 계산
        indices = image_indices.flatten() * pixels_per_image + pixel_coords.flatten()
        
        # RGB 데이터 추출
        matched_rgb = self.rgb_data[indices].reshape(H, W, 3)
        # matched_rgb = 1 - matched_rgb  # 기존 처리 유지
        
        return matched_rgb

    def get_matched_rgb_4(self, st_arr, output_fov):
        H, W = output_fov.shape[:2]
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        # 4개의 최근접 이웃 찾기
        st_distances, image_indices = self.find_nearest_st_arr_4(st_arr)  # shape: (H*W, 4)
        
        # 1개의 FOV 매칭 사용, 동일한 값을 4번 반복
        pixel_coords = self.match_output_to_data_fov_1(output_fov)       # shape: (H*W,)
        pixel_coords = np.repeat(pixel_coords, 4)                       # shape: (H*W * 4,)
        image_indices = image_indices.flatten()                          # shape: (H*W * 4,)
        st_distances = st_distances.flatten()                            # shape: (H*W * 4,)
        
        # 인덱스 계산
        indices = image_indices * pixels_per_image + pixel_coords       # shape: (H*W * 4,)
        
        # RGB 데이터 추출
        matched_rgb = self.rgb_data[indices].reshape(H * W, 4, 3)       # shape: (H*W, 4, 3)
        
        # 가중치 계산: 거리의 역수를 가중치로 사용
        weights = 1 / st_distances                                        # shape: (H*W * 4,)
        
        # 가중치 정규화: 각 픽셀마다 가중치의 합이 1이 되도록
        weights = weights.reshape(H * W, 4)
        weights /= weights.sum(axis=1, keepdims=True)                   # shape: (H*W, 4)
        
        # 가중 합산
        weighted_rgb = (matched_rgb * weights[..., np.newaxis]).sum(axis=1)  # shape: (H*W, 3)
        
        return weighted_rgb.reshape(H, W, 3)