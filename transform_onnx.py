import torch
import argparse
from src.model import Nerf4D_relu_ps
import os

def convert_to_onnx(args):
    # Load the model
    model = Nerf4D_relu_ps(D=args.mlp_depth, W=args.mlp_width, depth_branch=False)
    
    # Load the weights
    checkpoints_dir = f'/data/ysj/neulf_result/Exp_{args.exp_name}/checkpoints/'
    latest_checkpoint = max(os.listdir(checkpoints_dir), key=lambda x: int(x.split('-')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 4)  # Batch size 1, input size 4 (uvst)

    # Export the model
    torch.onnx.export(model, dummy_input, f"{args.exp_name}.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    print(f"Model converted and saved as {args.exp_name}.onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='', help='exp name')
    parser.add_argument('--mlp_depth', type=int, default=8)
    parser.add_argument('--mlp_width', type=int, default=256)
    args = parser.parse_args()

    convert_to_onnx(args)