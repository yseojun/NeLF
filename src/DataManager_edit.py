import numpy as np
import os

class DataManager:
    def __init__(self, base_path='/data/hmjung/data_backup/NeuLF_rgb/dataset/stanford_half/beans/', grid_size=17, image_size=(256, 512)):
        self.base_path = base_path

        self.image_size = image_size
        self.grid_size = grid_size
        
        self.uvst_data, self.rgb_data = self.load_data()
        
        self.preprocess()

    def load_data(self):
        print("데이터 로드 중...")
        uvst_train = np.load(os.path.join(self.base_path, 'uvsttrain.npy'))
        uvst_val = np.load(os.path.join(self.base_path, 'uvstval.npy'))
        rgb_train = np.load(os.path.join(self.base_path, 'rgbtrain.npy'))
        rgb_val = np.load(os.path.join(self.base_path, 'rgbval.npy'))

        uvst_data = np.concatenate((uvst_train, uvst_val), axis=0)
        rgb_data = np.concatenate((rgb_train, rgb_val), axis=0)

        self.uvst_val = uvst_val
        self.rgb_val = rgb_val

        print("데이터 로드 완료.")
        return uvst_data, rgb_data

    def preprocess(self):
        print("데이터 전처리 중...")
        num_images = self.grid_size * self.grid_size
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        st_values = self.uvst_data[::pixels_per_image, 2:4]
        
        t_sorted_indices = np.argsort(st_values[:, 1])
        
        # t값이 비슷한 그룹으로 나누기
        groups = np.array_split(t_sorted_indices, self.grid_size)
        
        final_indices = []
        for group in groups:
            group_st = st_values[group]
            # 각 그룹 내에서 s값으로 내림차순 정렬 (8 → -8)
            s_sorted_indices = np.argsort(group_st[:, 0])
            final_indices.extend(group[s_sorted_indices])
        
        # 정렬된 인덱스에 따라 데이터 재배열
        self.uvst_data = np.concatenate([self.uvst_data[i*pixels_per_image:(i+1)*pixels_per_image] 
                                       for i in final_indices])
        self.rgb_data = np.concatenate([self.rgb_data[i*pixels_per_image:(i+1)*pixels_per_image] 
                                      for i in final_indices])
        
        # 정렬 결과 출력
        print("\n정렬된 ST 값:")
        for i in range(num_images):
            print(f"Image {i:2d}: {self.uvst_data[i*pixels_per_image, 2:4] * 80}")
        
        print("데이터 전처리 완료.")

    def get_matched_rgb_4(self, st_arr):
        H, W = (512,512)
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        st_arr *= 80
        
        # 각 점에 대해 내림과 올림 값 계산
        s_floor = np.floor(st_arr[..., 0]).astype(int)
        s_ceil = np.ceil(st_arr[..., 0]).astype(int)
        t_floor = np.floor(st_arr[..., 1]).astype(int)
        t_ceil = np.ceil(st_arr[..., 1]).astype(int)
        
        corners_s = np.stack([s_floor, s_floor, s_ceil, s_ceil], axis=-1)
        corners_t = np.stack([t_floor, t_ceil, t_floor, t_ceil], axis=-1)
        
        st_distances = np.sqrt(
            (corners_s - st_arr[..., 0:1])**2 + 
            (corners_t - st_arr[..., 1:2])**2
        ) + 1e-6
        
        image_indices = (corners_t + 8) * self.grid_size + corners_s + 8
        image_indices = np.clip(image_indices, 0, self.grid_size * self.grid_size - 1)
        
        y_coords = np.arange(H).reshape(-1, 1)
        x_coords = np.arange(W).reshape(1, -1)
        pixel_coords = y_coords * W + x_coords
        pixel_coords = np.broadcast_to(pixel_coords.reshape(-1, 1), (H * W, 4))
        
        # 인덱스 계산
        indices = image_indices * pixels_per_image + pixel_coords
        
        # RGB 데이터 추출
        matched_rgb = self.rgb_data[indices].reshape(H * W, 4, 3)
        
        weights = 1 / st_distances                                  
        
        # 가중치 정규화
        weights = weights.reshape(H * W, 4)
        weights /= weights.sum(axis=1, keepdims=True)                 
        
        weighted_rgb = (matched_rgb * weights[..., np.newaxis]).sum(axis=1)
        
        return weighted_rgb.reshape(H, W, 3)

    def get_uvst_idx(self, image_idx):
        pixels_per_image = self.image_size[0] * self.image_size[1]
        
        uvst_data = self.uvst_data[image_idx * pixels_per_image:(image_idx + 1) * pixels_per_image]
        
        return uvst_data
    
    def get_uvst_val(self):
        uvst = self.uvst_data[0:262144]
        return uvst