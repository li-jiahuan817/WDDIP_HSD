import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

wv_list_hsd = np.array([420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
                        580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700]) * 1e-9
depth_list_hsd = [0.4515, 0.5685, 0.7155, 0.9, 1.133, 1.426, 1.7945]
depth_interval_hsd = [0.4, 0.503, 0.634, 0.797, 1.003, 1.263, 1.589, 2.0]

pixel_pitch = 1e-6
f_number = 1


def dst_calc(kernel_size):
    x, y = np.meshgrid(np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size),
                       np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size))
    x = x * pixel_pitch
    y = y * pixel_pitch
    dst_sq = np.sqrt(x ** 2 + y ** 2)
    return dst_sq

# https://refractiveindex.info/?shelf=other&book=Norland_NOA-61&page=Norland#google_vignette
def refractive_index_noa61(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


def gaussian_psf_generation(wv_list, depth_list, dst_sq, focal_length_550, focal_distance, material, scale):
    if material == 'noa':
        refractive_func = refractive_index_noa61
    sensor_distance = 1 / ((1 / focal_length_550) - (1 / focal_distance))
    aperture_size = focal_length_550 / f_number
    psfs = []
    for depth in depth_list:
        psf_d = []
        for wv in wv_list:
            focal_length = focal_length_550 * (refractive_func(550e-9) / refractive_func(wv))
            deviation = (1 / 2) * aperture_size * np.abs(((1 / focal_length) - (1 / depth)) * sensor_distance - 1) * scale
            if deviation == 0:
                deviation = 0.0001
            normal = 1 / (2 * np.pi * (deviation ** 2))
            gauss_psf = np.exp(-(dst_sq / (2 * (deviation ** 2)))) * normal
            gauss_psf = gauss_psf / np.sum(gauss_psf)
            psf_d.append(gauss_psf)
        psfs.append(np.stack(psf_d))

    return psfs


def visualize_psfs(psfs):
    fig, axes = plt.subplots(7, 15, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.suptitle("PSF", fontsize=10, y=0.95)
    for d in range(len(depth_list_hsd)):
        d_idx = d
        w_idx = 0
        for w in range(29):
            psf = psfs[d][w, :, :]
            if w % 2 != 0:
                continue

            axes[d_idx, w_idx].imshow(psf, cmap='gray')
            axes[d_idx, w_idx].axis('off')
            w_idx += 1

    # fig.savefig('psf.png')
    plt.show()


def generate_psfs(kernel_size, focal_length_550, focal_distance, material, scale):
    dst_sq = dst_calc(kernel_size)
    psfs = gaussian_psf_generation(wv_list_hsd, depth_list_hsd, dst_sq, focal_length_550, focal_distance, material, scale)

    return psfs


def depth_dep_convolution(img, psfs, depth_map, kernel_size, depth_interval=depth_interval_hsd):
    depth_map = depth_map * 1e-3
    img = torch.from_numpy(img)
    depth_map = torch.from_numpy(depth_map)

    zeros_tensor = torch.zeros_like(img, dtype=torch.double)

    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        depth_min = depth_interval[depth_idx]
        depth_max = depth_interval[depth_idx+1]
        mask_temp1 = depth_map >= depth_min
        mask_temp2 = depth_map < depth_max
        depth_mask = mask_temp1 * mask_temp2
        if torch.sum(depth_mask) == 0:
            continue

        psf = torch.from_numpy(psf)
        psf = F.pad(psf, [(img.size()[-1]-kernel_size)//2, (img.size()[-1]-kernel_size)//2 + 1, (img.size()[-2]-kernel_size)//2, (img.size()[-2]-kernel_size)//2 + 1])
        img_fft = torch.fft.fft2(img, dim=(-2, -1))
        psf_fft = torch.fft.fft2(psf, dim=(-2, -1))
        blurred_img = torch.fft.ifft2(img_fft * psf_fft)
        blurred_img = torch.fft.ifftshift(blurred_img, dim=(-2, -1)).real

        blurred_imgs.append(torch.where(depth_mask, blurred_img, zeros_tensor))

    return torch.sum(input=torch.stack(blurred_imgs), dim=0)


def add_noise(img, std):
    noise = np.random.normal(0, std, img.shape)
    img_noisy = img + noise
    img_noisy = img_noisy * (img_noisy >= 0)

    return img_noisy