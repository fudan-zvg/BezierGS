import torch
from argparse import ArgumentParser
from utils.sh_utils import SH2RGB
from utils.general_utils import save_ply

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='')
    args = parser.parse_args()

    pth_path = args.pth_path
    # load pth file
    model_args = torch.load(pth_path)
    gaussian_params = model_args[0]

    static_xyz = model_args[0][1][:, 0, :]
    static_rgb = SH2RGB(model_args[0][2][:, 0, :])

    opa_mask = torch.sigmoid(model_args[0][6][:, 0]) > 0.05
    save_ply(static_xyz[opa_mask], 'gaussian.ply', static_rgb[opa_mask])
