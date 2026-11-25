import math
from abc import ABC
import numpy as np
import tensorflow as tf

from log import Logger
from .util import complex_exponent_tf
from constants import lcos_phase_modulate_list


def summary_grayscale_map(grayscale_map, doe_tag):
    tf.summary.image(name="Grayscale_map Normalized_{}".format(doe_tag), data=grayscale_map)


def _copy_quad_to_full(quad_map):
    height_map_half_left = tf.concat([tf.reverse(quad_map, axis=[0]), quad_map], axis=0)
    height_map_full = tf.concat([tf.reverse(height_map_half_left, axis=[1]), height_map_half_left], axis=1)
    return height_map_full


class SimulatedDOELayer(ABC, tf.keras.layers.Layer):

    @staticmethod
    def shift_phase_according_to_grayscale_map(input_field, grayscale_map, wave_lengths,
                                               wavelength_to_refractive_index_func, doe_tag):

        """
        Args:
            doe_tag:
            grayscale_map: gray scale map loaded to SLM
            input_field: Input field.
            gray_map: Simulated DOE Layer with 0-255
            wave_lengths: 400-700nm wave lengths list resolution = 10nm
            wavelength_to_refractive_index_func:

        delta_phase = (2*pi/wave_length) * d * [ne(wave_length) - no(wave_length)]
        The formula does not take into account the angle between the incident wave and the optical axis

        Returns:Modulated wave field.
        """
        summary_grayscale_map(grayscale_map, doe_tag)
        unit_shift_phase = lcos_phase_modulate_list * np.pi / 255
        unit_shift_phase = unit_shift_phase.reshape([1, 1, 1, -1])
        # grayscale_map has been normalized
        delta_phase = grayscale_map * 255 * unit_shift_phase
        phase_shifts = complex_exponent_tf(delta_phase)
        phase_shifts = tf.cast(phase_shifts, tf.complex64)
        input_field = tf.cast(input_field, tf.complex64)
        # phase_shifts.shape=(1,1080,1920,31)->(1,1920,1920,31)
        phase_shifts = tf.pad(tensor=phase_shifts,
                              paddings=[[0, 0], [420, 420],
                                        [0, 0], [0, 0]])
        shifted_field = tf.multiply(input_field, phase_shifts, name='phase_plate_shift')
        return shifted_field

    @staticmethod
    def add_grayscale_map_noise(grayscale_map, tolerance=None):
        """

        Args:
            grayscale_map:
            tolerance:

        Returns:

        """
        if tolerance is not None:
            grayscale_map = grayscale_map + tf.random.uniform(shape=grayscale_map.shape,
                                                              minval=-tolerance,
                                                              maxval=tolerance, dtype=grayscale_map.dtype)
            Logger.i("Simulated fabrication noise on height map: %0.2e" % tolerance)
            return grayscale_map

    def preprocess_grayscale_map(self, training=None, testing=None):

        pass

    def modulate(self, input_field, preprocess_grayscale_map, grayscale_map_regularizer, grayscale_tolerance,
                 wave_length_list, wavelength_to_refractive_index_func, doe_tag):
        if grayscale_map_regularizer is not None:
            self.add_loss(grayscale_map_regularizer(preprocess_grayscale_map))
            preprocess_grayscale_map = self.add_grayscale_map_noise(preprocess_grayscale_map,
                                                                    tolerance=grayscale_tolerance)
        return self.shift_phase_according_to_grayscale_map(input_field=input_field,
                                                           grayscale_map=preprocess_grayscale_map,
                                                           wave_lengths=wave_length_list,
                                                           wavelength_to_refractive_index_func=wavelength_to_refractive_index_func,
                                                           doe_tag=doe_tag)


def read_pretrained_grayscale_map(check_point_path):
    check_point = tf.train.latest_checkpoint(check_point_path)
    doe_var = tf.train.load_variable(check_point,
                                     "doe_layer/weight_height_map_radius_1d/.ATTRIBUTES/VARIABLE_VALUE")


class SimulatedHeightMapDOELayer(SimulatedDOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, block_size=1,
                 grayscale_map_initializer=None, grayscale_map_regularizer=None, grayscale_tolerance=None,
                 name="HeightMapDOELayer"):
        super(SimulatedHeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.block_size = block_size
        self.grayscale_map_initializer = grayscale_map_initializer
        self.grayscale_map_regularizer = grayscale_map_regularizer
        self.grayscale_tolerance = grayscale_tolerance
        self.weight_grayscale_map = None
        self.grayscale_map_shape = None
        self.gray_value_constraint_max = 255
        self.quantization_base_gray_value = 255
        self.step_counter = None

    def generalizable_pre_build(self):

        self.step_counter = self.add_weight(name="step_counter", shape=None, dtype=tf.int32, trainable=False,
                                            initializer=tf.keras.initializers.constant(value=0))

        if self.grayscale_map_initializer is None:
            init_grayscale_map_value = np.ones(shape=self.grayscale_map_shape, dtype=np.float32) * 5e-1
            self.grayscale_map_initializer = tf.keras.initializers.constant(value=init_grayscale_map_value)

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.grayscale_map_shape = [1, height, width, 1]
        self.generalizable_pre_build()
        self.weight_grayscale_map = self.add_weight(name="weight_grayscale_map",
                                                    shape=self.grayscale_map_shape,
                                                    dtype=tf.float32,
                                                    trainable=True,
                                                    initializer=self.grayscale_map_initializer,
                                                    constraint=lambda x: tf.clip_by_value(x, 0, 1)
                                                    )

    def preprocess_grayscale_map(self, training=None, testing=None):
        @tf.custom_gradient
        def _round_keep_gradients(_x):
            def _grad(_dy):
                return _dy

            return tf.round(_x), _grad

        def _full_precise_path(_weight):
            return _weight

        def _base_plane_wrapper(_etching_height_map_weight):
            # return self.base_plane_thickness - (self.quantization_base_height * _etching_height_map_weight)
            return _etching_height_map_weight

        final_processed_grayscale_map = self.weight_grayscale_map
        # !=== QE Record
        ideal_cond = _base_plane_wrapper(_full_precise_path(self.weight_grayscale_map))
        quantization_error_mae = tf.reduce_mean(tf.abs(ideal_cond - final_processed_grayscale_map))
        quantization_error_mse = tf.reduce_mean(tf.square(ideal_cond - final_processed_grayscale_map))
        tf.summary.scalar(name="quantization_error_mae", data=quantization_error_mae)
        tf.summary.scalar(name="quantization_error_mse", data=quantization_error_mse)
        tf.summary.scalar(name="step_counter", data=self.step_counter)
        if training:
            self.step_counter.assign_add(1)  # increase step

        return final_processed_grayscale_map

    def call(self, inputs, doe_tag=0, training=None, testing=None, **kwargs):
        grayscale_map = self.preprocess_grayscale_map(training=training, testing=testing)
        return self.modulate(input_field=inputs,
                             preprocess_grayscale_map=grayscale_map,
                             grayscale_map_regularizer=self.grayscale_map_regularizer,
                             grayscale_tolerance=self.grayscale_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func,
                             doe_tag=doe_tag)


class SimulatedRotationallySymmetricSimulatedHeightMapDOELayer(SimulatedHeightMapDOELayer):

    def build(self, input_shape):
        # _, height, width, _ = input_shape

        # self.grayscale_map_shape = int(height / 2)
        self.grayscale_map_shape = (2202 // 2)
        self.generalizable_pre_build()

        self.weight_grayscale_map = self.add_weight(name="weight_grayscale_map_radius_1d",
                                                    shape=self.grayscale_map_shape,
                                                    dtype=tf.float32,
                                                    trainable=True,
                                                    initializer=self.grayscale_map_initializer,
                                                    constraint=lambda x: tf.clip_by_value(x, 0, 1))

    def preprocess_grayscale_map(self, training=None, testing=None):
        grayscale_map_1d = super().preprocess_grayscale_map(training=training, testing=testing)
        radius = self.grayscale_map_shape
        # diameter = 2 * radius
        height, width = (1080, 1920)
        [x, y] = np.mgrid[0:height // 2, 0:width // 2].astype(np.float32)
        radius_distance = tf.sqrt(x ** 2 + y ** 2)
        # Radial selection
        grayscale_map_quad = tf.where(tf.logical_and(tf.less(radius_distance, 1.0),
                                                     tf.less_equal(0.0, radius_distance)),
                                      grayscale_map_1d[0], 0.0)

        for r in range(1, radius - 1):
            # grayscale_map_1d[r]
            grayscale_map_quad += tf.where(tf.logical_and(tf.less(radius_distance, tf.cast(r + 1, dtype=tf.float32)),
                                                          tf.less_equal(tf.cast(r, dtype=tf.float32), radius_distance)),
                                           grayscale_map_1d[r], 0.0)
        grayscale_map_full = _copy_quad_to_full(grayscale_map_quad)
        # grayscale_map = tf.reshape(grayscale_map_full, shape=[1, diameter, diameter, 1])
        grayscale_map = tf.reshape(grayscale_map_full, shape=[1, 1080, 1920, 1])
        return grayscale_map
