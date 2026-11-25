import numpy as np
import tensorflow as tf

from .util import transpose_2d_fft, transpose_2d_ifft, complex_exponent_tf, fft_shift_2d_tf, ifft_shift_2d_tf
from .diffractive_optical_element import DOELayer
from .camera import circular_aperture


# def lens_transfer_function(_x, _y, _k, len_focal):
#     '''
#     :param _x:
#     :param _y:
#     :param len_focal: focal length of len
#     :return: complex_shift_phase
#     '''
#     _r = np.sqrt(_x ** 2 + _y ** 2)
#     shift_phase = _k * _r ** 2 / (2 * len_focal)
#     complex_shift_phase = np.exp(-1j * shift_phase)
#     return complex_shift_phase


# 此处修改传播过程

# 计算汇聚透镜对平面波的相位调制


class FresnelPropagation0(tf.keras.layers.Layer):
    def __init__(self, wave_lengths, doe_layer: DOELayer = None):
        super().__init__()

        self.wave_lengths = wave_lengths.reshape([1, 1, 1, -1])
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = 4.8e-6
        self.x = None
        self.y = None
        self.fx = None
        self.fy = None
        self.focal_length = 50e-3
        self.lens_diameter = 25e-3
        self.circle_aperture = None
        self.len_phase_factor = None
        self.fre_factor = None
        self.fre_factor2 = None
        # self.rect_aperture = None
        # self.field_after_height_map = None

        self.doe_layer = doe_layer

    def lens_transfer_function(self, _x, _y, _k, len_focal):
        _r = np.sqrt(_x ** 2 + _y ** 2)
        shift_phase = _k * _r ** 2 / (2 * len_focal)
        # complex_shift_phase = np.exp(-1j * shift_phase)
        complex_shift_phase = complex_exponent_tf(-1 * shift_phase)
        return complex_shift_phase

    def circle_function(self, _x, _y, aperture_diameter):
        '''
        :param size:
        :param rect_size_ratio:
        :return:
        '''
        _r = np.sqrt(_x ** 2 + _y ** 2)
        r_size = np.zeros_like(_x)
        r_size[_r <= aperture_diameter / 2] = 1
        r_size = tf.convert_to_tensor(r_size)
        return r_size  # core resizing

    # 矩孔函数 有量纲
    def rect_function(self, _x, _y, aperture_diameter):
        '''
        :param size:
        :param rect_size_ratio:
        :return:
        '''
        r_size = np.zeros_like(_x)
        r_size[(np.abs(_x) <= aperture_diameter / 2) & (np.abs(_y) <= aperture_diameter / 2)] = 1
        r_size = tf.convert_to_tensor(r_size)
        return r_size  # core resizing

    def build(self, input_shape):
        batch_num, m_original, n_original, channel_num = input_shape
        # zero padding
        m_padding = m_original // 4
        n_padding = n_original // 4
        self.m_padding = m_padding
        self.n_padding = n_padding
        # m_full = m_original + 2 * m_padding
        # n_full = n_original + 2 * n_padding
        m_full, n_full = (1080, 1920)
        [x1, y1] = np.mgrid[-m_full // 2:m_full // 2, -n_full // 2:n_full // 2]
        self.x = x1 * self.discretization_size
        self.y = y1 * self.discretization_size
        # Spatial frequency
        # 空域与频域抽样数量（M,N）相同
        self.fx = x1 / (self.discretization_size * m_full)  # max frequency = 1 /(2*pixel_size)
        self.fy = y1 / (self.discretization_size * n_full)
        # np.fftshift(x) [, axes] ）将零频率分量移到频谱中心。
        self.x = self.x[None, :, :, None]
        self.y = self.y[None, :, :, None]
        self.fx = self.fx[None, :, :, None]
        self.fy = self.fy[None, :, :, None]

        self.len_phase_factor = self.lens_transfer_function(self.x, self.y, self.wave_nos, self.focal_length)
        # self.circle_aperture = self.circle_function(self.x, self.y, self.lens_diameter)
        # self.rect_aperture = self.rect_function(self.x, self.y, self.lens_diameter)
        self.fre_factor = complex_exponent_tf(self.wave_nos * self.focal_length *
                                              tf.math.sqrt(1 - (self.wave_lengths * self.fx) ** 2 - (
                                                      self.wave_lengths * self.fy) ** 2))
        self.fre_factor2 = complex_exponent_tf(self.wave_nos * (self.focal_length - 20e-3) *
                                               tf.math.sqrt(1 - (self.wave_lengths * self.fx) ** 2 - (
                                                       self.wave_lengths * self.fy) ** 2))

    def _propagate(self, input_field):
        # distance_z0 = self.focal_length
        # 模拟点光源->球面波自由传播 f mm ->U0
        # U0 = generate_spherical_wave(self.x, self.y, 0, 0, self.focal_length, self.wave_lengths)
        # print(f'input_field.shape ={input_field}')
        # U0 = input_field
        # 透射len1 相位变换  U0 * len1_phase_factor -> U1
        # U0.shape = (1, 1080, 1920, 31)
        # self.len_phase_factor.shape = (1, 1080, 1920, 31)
        # U0 = tf.pad(tensor=input_field,
        #             paddings=[[0, 0], [self.m_padding, self.m_padding],
        #                       [self.n_padding, self.n_padding], [0, 0]])
        U1 = input_field * self.len_phase_factor

        # 透射len1 孔径光阑(光瞳函数模拟) F1(u,v) =fft{U1}  *  OTF1-> F2
        # 对孔径函数 fft -> OTF1()
        F1 = fft_shift_2d_tf(transpose_2d_fft(U1))
        # lowpass_filter1 = fft_shift_2d_tf(transpose_2d_fft(self.circle_aperture))
        # F2 = F1 * 1  # lowpass_filter1
        # # fresnel传播1 distance_z1
        # # F2 * np.exp(1j * k * distance_z * np.sqrt(1 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2))-> F3
        # self.fre_factor.shape = (1, 1080, 1920, 31)
        F3 = F1 * self.fre_factor

        # 透射 LCOS 孔径光阑  F3 * OTF_lcos-> F4
        # lowpass_filter2 = tf.signal.fftshift(transpose_2d_fft(self.rect_aperture), axes=(-3, -2))
        # F4 = F3 * 1  # lowpass_filter2

        # U5 = transpose_2d_ifft(fourier_padded_input_field)
        U4 = transpose_2d_ifft(ifft_shift_2d_tf(F3))
        # self.field_after_height_map = self.doe_layer(U4, training=True, testing=False)
        ## self.field_after_height_map = self.doe_layer(U4, training=True, testing=False)
        # U4 = tf.slice(U4, [0, 256, 256, 0], [1, 1024, 1024, 31])
        self.field_after_height_map = self.doe_layer(U4, training=True, testing=False)
        # self.field_after_height_map = tf.pad(tensor=self.field_after_height_map,
        #                                      paddings=[[0, 0], [self.m_padding, self.m_padding],
        #                                                [self.n_padding, self.n_padding], [0, 0]])
        # 1080 * 1920
        # U5 = self.field_after_height_map
        # fresnel传播2 distance_z2  F5=fft{U5} * np.exp(B) -> F6
        F6 = fft_shift_2d_tf(transpose_2d_fft(self.field_after_height_map)) * self.fre_factor
        # 透射len2 孔径光阑  F6 * OTF2-> F7
        # lowpass_filter3 = lowpass_filter1
        # F7 = F6 * 1  # lowpass_filter3
        # 透射len2 相位变换  U7=ifft{F7} * len2_phase_factor -> U8
        U8 = transpose_2d_ifft(ifft_shift_2d_tf(F6)) * self.len_phase_factor
        F8 = fft_shift_2d_tf(transpose_2d_fft(U8))
        # fresnel传播3      F8=fft{U8} * np.exp(C)-> F9
        F9 = F8 * self.fre_factor2
        # U9 <- np.fft.ifft2{F9}
        output_field = transpose_2d_ifft(ifft_shift_2d_tf(F9))  #
        print(f'output_field.shape = {output_field.shape}')
        # tf.summary.scalar(name="out_field_abs_sum", data=tf.reduce_sum(tf.cast(tf.abs(output_field), dtype=tf.float32)))
        # return output_field[:, self.m_padding:-self.m_padding, self.n_padding:-self.n_padding, :]
        return output_field
    def call(self, inputs, **kwargs):
        return self._propagate(input_field=inputs)
