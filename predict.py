import rasterio
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import cv2
from models.gans import Generator
from utils.utils import plot_image, intersect_dicts


def predict(opt):   
    # model  
    model = Generator()

    # GPU-support
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load weights
    assert os.path.exists(opt.weight), "no found the model weight"
    checkpoint = torch.load(opt.weight)
    
    csd = checkpoint['G_XY'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict())
    model.load_state_dict(csd, strict=False)

    # input
    img = open_geotiff(opt.input, opt.band)
    img = normalize_img(img)
        
    img = torch.from_numpy(img).float() # tensor in [0,1]
    imgs = img.unsqueeze(0).unsqueeze(0) # (1, 3, H, W)
    
    print('predicting...')
    model.eval()
    imgs.to(device)
    with torch.no_grad():
        preds = model(imgs)  # (1, C, H, W)
        fake_img = preds[0].mul(16383).clamp(0,16383).cpu().detach().numpy().squeeze(0).astype(np.uint16) # (1, H, W)
        save_file = os.path.join('outputs', os.path.basename(opt.input).replace('.tif', '_fake.tif'))
        cv2.imwrite(save_file, fake_img)
    print('done')

def open_geotiff(path, band=1):
    with rasterio.open(path) as src:
        data = src.read(band)
    assert data.all() != None, f'{path} is not found'
    data = data.astype(np.uint16)
    return data

def normalize_img(img):
    hr_bits = 11
    img = img.astype(np.float32)/(2**hr_bits - 1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default = 'data/wv3-k3a/train/WV3/SN3_roads_train_AOI_4_Shanghai_PAN_img37.tif', help='input image')
    parser.add_argument('--band', '-b', type=int, default=1, help='band of input image')
    parser.add_argument('--weight', '-w', default='results/weight/jbnu_imgtrans_best.pth',
                        help='weight file path')
    opt = parser.parse_args()

    predict(opt)