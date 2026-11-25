import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# test1
# 图像升维操作

# # array_a = np.arange(10)
# array_a = np.random.randint(10, size=12)
# print(f'array_a = {array_a}')
# array_a = array_a.reshape(3, 4)
# print(f'array_a = {array_a}')
# # 最外层升维
# array_b = array_a[None, :, :]
# print(f'array_b = {array_b}')
# # 最内层升维
# array_c = array_a[:, :, None]
# print(f'array_c.shape = {array_c.shape}')
# print(f'array_c = {array_c}')
# # 最内\外层双升维
# array_d = array_a[None, :, :, None]
# # array_d.shape = (1, 3, 4, 1)
# print(f'array_d.shape = {array_d.shape}')

# test2

# 测试 "：" 与 "..." 的区别
# array_e = array_a[:, None]
# print(f'array_e.shape = {array_e.shape}')

# test3

# central_wavelength = 520
# central_wavelength = central_wavelength * 1e-9
# from constants import refractive_index_liquid_crystal_e7
#
# wave_number = 2 * np.pi / central_wavelength
# delta_phase = 0
# delta_n = refractive_index_liquid_crystal_e7(central_wavelength)
# print(f'delta_n = {delta_n}')
# with_phase = False
# with_grayscale_map = True
# if with_phase:
#     lc_thickness = delta_phase / (wave_number * delta_n)
# grayscale_map = np.ones(shape=(1024, 1024), dtype=np.float32) * 255
# if with_grayscale_map:
#     unit_shift_phase = 2 * np.pi / 255
#     delta_phase_with_grayscale_map = grayscale_map * unit_shift_phase
#     print(f'delta_phase_with_grayscale_map = {delta_phase_with_grayscale_map}')
#     delta_phase = delta_phase_with_grayscale_map
#
#     lc_thickness = delta_phase / (wave_number * delta_n)
#
# print(f'lc_thickness.shape = {lc_thickness.shape}')


# test4
# from constants import refractive_index_glass_bk7
# from constants import wave_length_list_400_700nm
# bk7_thickness = 25 * 1e-3
# optical_path = bk7_thickness * refractive_index_glass_bk7(wave_length_list_400_700nm)
# distance_with_lens_and_slm = optical_path + 63 * 1e-3
# print(f'optical_path = {distance_with_lens_and_slm}')


# test5
#  tf.signal.ifftshift 与 np.fft.ifftshift功能测试

# fx = np.arange(16).reshape(4, 4)
# ifx = tf.signal.ifftshift(fx)
# ifx2 = np.fft.ifftshift(fx)
# print(f'fx = {fx}')
# print(f'ifx = {ifx}')
# print(f'ifx2 = {ifx2}')


# test 6
# test_array = np.arange(12).reshape(3,4)
# print(f'test_array:{test_array}')
# print(f'test_array[0]:{test_array[0]}')

# test7
# array_a = np.ones(shape=(1024, 1024), dtype=np.float32)
# array_b = np.ones(shape=(512, 512), dtype=np.float32)
# height, width = array_a.shape
# radius = int(height / 2)
# diameter = 2 * radius
# [x, y] = np.mgrid[0:diameter // 2, 0:diameter // 2].astype(np.float32)
#
# radius_distance = tf.sqrt(x ** 2 + y ** 2)
# print(f'x.shape={x.shape}')
# print(f'y.shape={y.shape}')
# print(f'radius_distance.shape={radius_distance.shape}')
# print(f'array_b[0].shape={array_b[0].shape}')
# print(f'array_b[1].shape={array_b[1].shape}')
#
# array_b = tf.where(tf.logical_and(tf.less(radius_distance, 3.0),
#                                   tf.less_equal(1.0, radius_distance)),
#                    array_b[0], 0.0)
# print(f'array_b = {array_b}')
# print(f'array_b.shape = {array_b.shape}')

# test 8
# psf = np.random.randint(9,size=(3, 3))
# print(f'psf = {psf}')
# psf = psf.reshape(1, 3, 3, 1)
#
# psf = psf.transpose(1, 2, 0, 3)
# psfs = []
# print(f'psf = {psf}')
# for r in range(3):
#     t = r
#     psfs.append(psf)
# print(f't = {t}')
# print(f'psfs = {psfs}')
# print(f'psf.shape = {psf.shape}')

# test 9
# 测试FFT
# 将绘制的图显示在窗口
# %matplotlib qt5

#
# img = cv2.imread(r"test_rgb/image.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 傅里叶变换
# f = np.fft.fft2(img)
#
# # 移动中心位置
# fshift = np.fft.fftshift(f)
#
# # 调整值范围
# result = 20 * np.log(np.abs(fshift))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.title("original")
# plt.axis("off")
#
# plt.subplot(1, 2, 2)
# plt.imshow(result, cmap=plt.cm.gray)
# plt.title("result")
# plt.axis("off")
#
# plt.show()


# test 10
# 测试 复指数E =A * exp(i*θ)

# phase = complex(2,3)
# A = 5
# A = complex(A)
#
# E = A * phase
# print(f"E = {E}")

# test 11
# 测试tensor变量的numpy()函数
# array_a = np.arange(12).reshape(3, 4)
# tf_a = tf.convert_to_tensor(array_a)
# print(f'tf_a.type = {type(tf_a)}')
# conv_array_a = tf_a.numpy()
# print(conv_array_a)
a = tf.constant(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
b = a
c = a
d = tf.concat([a[:,..., 0,None], b[:, ..., 1,None], c[:, ..., 2,None]], axis=-1)
e = a[:, ..., 0,None]
print(f"a.shape = {a.shape}")
print(f"d.shape = {d.shape}")
print(f"e.shape = {e.shape}")
print(f"e = {e}")
print(f"d = {d}")

