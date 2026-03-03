from optics_utils import generate_psfs, visualize_psfs, depth_dep_convolution, add_noise
import glob
import os
import numpy as np

hs_gt_root = 'Your DIRECTORY to save GT HSI patch'

# You can generate degraded HSIs with different parameters
focal_length = [16e-3]
focused_distance = [1.2]
scale = 20 # scaling factor
kernel_size = 25

for i in range(len(focal_length)):
    focal_length_i = focal_length[i]
    focused_distance_i = focused_distance[i]
    opt_name = str(int(focal_length_i*1e3)) + '_' + str(int(focused_distance_i*1e2)) + '_' + str(kernel_size)
    psfs = generate_psfs(kernel_size, focal_length_i, focused_distance_i, 'noa', scale)
    # visualize_psfs(psfs)
    os.makedirs('Your DIRECTORY to save degraded HSI patch/'+opt_name, exist_ok=True)
    files_source = glob.glob(os.path.join(hs_gt_root, '*.npy'))
    files_source.sort()
    for f in files_source:
        imgname = os.path.basename(f)
        hs_cube = np.load(f)
        depth_cube = np.load('Your DIRECTORY to save GT depth patch/'+imgname)
        imgname = os.path.splitext(imgname)[0]
        hs_blurred = depth_dep_convolution(hs_cube, psfs, depth_cube, kernel_size)
        hs_processed = add_noise(hs_blurred.numpy(), std=0.01)
        np.save('Your DIRECTORY to save degraded HSI patch/'+opt_name+'/'+imgname+'.npy', hs_processed)