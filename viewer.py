from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import os
import torch
import numpy as np
import glob
import time
import imageio

from src.model import Nerf4D_relu_ps
from src.utils import rm_folder, rm_folder_keep
import torchvision

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

image_folder = "static/image"
time_array = []
current_image_index = 0
theta = np.pi
theta_x, theta_y, theta_z = 0, 0, 0

# 기본값 설정
default_config = {
    'exp_name': '03_9to14_d4w256_e121',
    'gpuid': '0',
    'mlp_depth': 4,
    'mlp_width': 256,
    'scale': 1,
    'img_form': '.png'
}

class ImageGenerator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_model(config)

    def load_model(self, config=None):
        if config:
            self.config.update(config)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpuid']
        print('>>> Using GPU: {}'.format(self.config['gpuid']))

        model = Nerf4D_relu_ps(D=self.config['mlp_depth'], W=self.config['mlp_width'], depth_branch=False)
        exp = 'Exp_' + self.config['exp_name']
        checkpoints = 'result/' + exp + '/checkpoints/'

        ckpt_paths = glob.glob(checkpoints + "*.pth")
        iter_count = 0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                iter_count = max(iter_count, ckpt_id)
            ckpt_name = f"./{checkpoints}/nelf-{iter_count}.pth"
        print(f"Load from {ckpt_name}")
        ckpt = torch.load(ckpt_name)
        print(ckpt)
        model.load_state_dict(ckpt)
        self.model = model.cuda()

    def get_vec3_xyz_torch(self, x, y, z, theta_x, theta_y, theta_z, H=360, W=360, device='cuda'):
        aspect = W / H
        theta_x = torch.tensor(theta_x, device=device)
        theta_y = torch.tensor(theta_y, device=device)
        theta_z = torch.tensor(theta_z, device=device)

        cos_theta_x = torch.cos(theta_x)
        sin_theta_x = torch.sin(theta_x)
        cos_theta_y = torch.cos(theta_y)
        sin_theta_y = torch.sin(theta_y)
        cos_theta_z = torch.cos(theta_z)
        sin_theta_z = torch.sin(theta_z)

        rot_x = torch.tensor([
            [1, 0, 0],
            [0, cos_theta_x, -sin_theta_x],
            [0, sin_theta_x, cos_theta_x]
        ], device=device)

        rot_y = torch.tensor([
            [cos_theta_y, 0, sin_theta_y],
            [0, 1, 0],
            [-sin_theta_y, 0, cos_theta_y]
        ], device=device)

        rot_z = torch.tensor([
            [cos_theta_z, -sin_theta_z, 0],
            [sin_theta_z, cos_theta_z, 0],
            [0, 0, 1]
        ], device=device)

        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(1, -1, H, device=device) / aspect

        u, v = torch.meshgrid(u, v, indexing='xy')
        dirs = torch.stack((u, v, -torch.ones_like(u)), dim=-1)

        rot_matrix = rot_z @ rot_y @ rot_x
        dirs = dirs @ rot_matrix.T
        dirs = dirs.reshape(-1, 3)

        tx = torch.full_like(dirs[:, 0:1], x)
        ty = torch.full_like(dirs[:, 0:1], y)
        tz = torch.full_like(dirs[:, 0:1], z)

        vec3_xyz = torch.cat((dirs, tx, ty, tz), dim=1)
        vec3_xyz = vec3_xyz.reshape(-1, 6)

        return vec3_xyz

    def generate_image(self, vec3_xyz, save_path, width=360, height=360):
        print('Generate image called')
        vec3_xyz = vec3_xyz.cuda()
        pred_color = self.model(vec3_xyz)
        pred_img = pred_color.reshape((width, height, 3)).permute((2, 0, 1))
        torchvision.utils.save_image(pred_img, save_path)
        


@app.route("/static/<path:filename>")
def static_file(filename):
    response = send_from_directory('static', filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def home():
    return render_template('home.html')

@socketio.on('request_new_image')
def handle_request_new_image(data):
    generator = app.config['generator']

    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    z = float(data.get('z', 0))
    roll = float(data.get('roll', 0))
    pitch = float(data.get('pitch', 0))
    size = 720
    height, width = size, size

    save_path = 'static/generated_image.png'

    start = time.time()
    vec3_xyz = generator.get_vec3_xyz_torch(x, y, z, roll, pitch, np.pi, height, width)
    generator.generate_image(vec3_xyz, save_path, width, height)
    end = time.time()
    time_val = end - start
    time_array.append(time_val)
    print(f"Inference time: {time_val} sec")
    print(f"Average inference time: {np.mean(time_array)} sec")
    socketio.emit('new_image', {'image_file': 'generated_image.png', 'time': time_val, 'avg_time': np.mean(time_array)})

if __name__ == "__main__":
    app.config['generator'] = ImageGenerator(default_config)
    socketio.run(app, debug=True, host='0.0.0.0', port=6006)
