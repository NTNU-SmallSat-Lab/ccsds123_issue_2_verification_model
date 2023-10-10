from . import header as hd
from . import constants as const
from .utils import clip, sign, sign_positive, modulo_star
import numpy as np

class Predictor():
    """
    CCSDS 123.0-B-2 high level model of the predictor
    """
    header = None
    image_constants = None
    image_sample = None # Symbol: s

    def __init__(self, header, image_constants, image_sample):
        self.header = header
        self.image_constants = image_constants
        self.image_sample = image_sample

    # Predictor constants
    local_difference_values_num = None

    spectral_bands_used = None # Symbol: P^*. Indexed by z
    # spectral_bands_used_mask = None

    weight_component_resolution = None # Symbol: Omega
    weight_update_change_interval = None # Symbol: t_inc
    weight_update_initial_parameter = None # Symbol: nu_min
    weight_update_final_parameter = None # Symbol: nu_max
    weight_update_scaling_exponent = None # Symbol: rho. Indexed by t
    weight_min = None # Symbol: omega_min
    weight_max = None # Symbol: omega_max

    register_size = None # Symbol: R

    def __init_predictor_constants(self):
        self.local_difference_values_num = self.header.prediction_bands_num
        if self.header.prediction_mode == hd.PredictionMode.FULL:
            self.local_difference_values_num += 3

        self.spectral_bands_used = np.empty((self.header.z_size), dtype=np.int64)
        for z in range(self.header.z_size):
            self.spectral_bands_used[z] = min(z, self.header.prediction_bands_num)

        self.weight_component_resolution = self.header.weight_component_resolution + 4
        self.weight_update_change_interval = 2**self.header.weight_update_change_interval
        self.weight_update_initial_parameter = self.header.weight_update_initial_parameter - 6
        self.weight_update_final_parameter = self.header.weight_update_final_parameter - 6
        self.weight_update_scaling_exponent = np.empty((self.header.y_size * self.header.x_size), dtype=np.int64)
        for t in range (self.header.y_size * self.header.x_size):
            self.weight_update_scaling_exponent[t] = clip( \
                self.weight_update_initial_parameter + int((t - self.header.x_size) / self.weight_update_change_interval) \
                , self.weight_update_initial_parameter, self.weight_update_final_parameter) \
                + self.image_constants.dynamic_range_bits - self.weight_component_resolution
        self.weight_min = -2**(self.weight_component_resolution + 2)
        self.weight_max = 2**(self.weight_component_resolution + 2) - 1

        self.register_size = self.header.register_size
        if self.register_size == 0:
            self.register_size = 64


    # Predictor variables
    local_sum = None # Symbol: sigma
    local_difference_vector = None # Symbol: U-vector
    weight_vector = None # Symbol: W-vector
    predicted_central_local_difference = None # Symbol: d
    high_resolution_predicted_sample_value = None # Symbol: s-breve
    double_resolution_predicted_sample_value = None # Symbol: s-tilde
    predicted_sample_value = None # Symbol: s-hat
    prediction_residual = None # Symbol: delta
    maximum_error = None # Symbol: m
    quantizer_index = None # Symbol: q
    clipped_quantizer_bin_center = None # Symbol: s'
    double_resolution_sample_representative = None # Symbol: s''-hat
    sample_representative = None # Symbol: s''
    double_resolution_prediction_error = None # Symbol: e
    scaled_prediction_endpoint_difference = None # Symbol: theta
    mapped_quantizer_index = None # Symbol: delta
    
    def __init_predictor_arrays(self):
        image_shape = self.image_sample.shape
        local_difference_vector_shape = image_shape + (self.local_difference_values_num,) 

        value = -1

        self.local_sum = np.full(image_shape, value, dtype=np.int64)
        self.local_difference_vector = np.zeros(local_difference_vector_shape, dtype=np.int64)
        self.weight_vector = np.zeros(local_difference_vector_shape, dtype=np.int64)
        self.predicted_central_local_difference = np.full(image_shape, value, dtype=np.int64)
        self.high_resolution_predicted_sample_value = np.full(image_shape, value, dtype=np.int64)
        self.double_resolution_predicted_sample_value = np.full(image_shape, value, dtype=np.int64)
        self.predicted_sample_value = np.full(image_shape, value, dtype=np.int64)
        self.prediction_residual = np.full(image_shape, value, dtype=np.int64)
        self.maximum_error = np.full(image_shape, value, dtype=np.int64)
        self.quantizer_index = np.full(image_shape, value, dtype=np.int64)
        self.clipped_quantizer_bin_center = np.full(image_shape, value, dtype=np.int64)
        self.double_resolution_sample_representative = np.full(image_shape, value, dtype=np.int64)
        self.sample_representative = np.full(image_shape, value, dtype=np.int64)
        self.double_resolution_prediction_error = np.full(image_shape, value, dtype=np.int64)
        self.scaled_prediction_endpoint_difference = np.full(image_shape, value, dtype=np.int64)
        self.mapped_quantizer_index = np.full(image_shape, value, dtype=np.int64)

    
    def __calculate_local_sum(self, x, y, z, t):
        if t == 0:
            return
        
        if self.header.local_sum_type == hd.LocalSumType.WIDE_NEIGHBOR_ORIENTED:
            if y > 0 and 0 < x and x < self.header.x_size - 1:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z] + \
                    self.sample_representative[y - 1, x - 1, z] + \
                    self.sample_representative[y - 1, x    , z] + \
                    self.sample_representative[y - 1, x + 1, z]
            elif y == 0 and x > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z] * 4
            elif y > 0 and x == 0:
                self.local_sum[y, x, z] = ( \
                    self.sample_representative[y - 1, x    , z] + \
                    self.sample_representative[y - 1, x + 1, z]) * 2
            elif y > 0 and x == self.header.x_size - 1:
                 self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z] + \
                    self.sample_representative[y - 1, x - 1, z] + \
                    self.sample_representative[y - 1, x    , z] * 2
                
        elif self.header.local_sum_type == hd.LocalSumType.NARROW_NEIGHBOR_ORIENTED:
            if y > 0 and 0 < x and x < self.header.x_size - 1:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y - 1, x - 1, z] + \
                    self.sample_representative[y - 1, x    , z] * 2 + \
                    self.sample_representative[y - 1, x + 1, z]
            elif y == 0 and x > 0 and z > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z - 1] * 4
            elif y > 0 and x == 0:
                self.local_sum[y, x, z] = ( \
                    self.sample_representative[y - 1, x    , z] + \
                    self.sample_representative[y - 1, x + 1, z]) * 2
            elif y > 0 and x == self.header.x_size - 1:
                 self.local_sum[y, x, z] = ( \
                    self.sample_representative[y - 1, x - 1, z] + \
                    self.sample_representative[y - 1, x    , z]) * 2
            elif y == 0 and x > 0 and z == 0:
                self.local_sum[y, x, z] = \
                    self.image_constants.middle_sample_value * 4

        elif self.header.local_sum_type == hd.LocalSumType.WIDE_COLUMN_ORIENTED:
            if y > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y - 1, x    , z] * 4
            elif y == 0 and x > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z] * 4
                
        elif self.header.local_sum_type == hd.LocalSumType.NARROW_COLUMN_ORIENTED:
            if y > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y - 1, x    , z] * 4
            elif y == 0 and x > 0 and z > 0:
                self.local_sum[y, x, z] = \
                    self.sample_representative[y    , x - 1, z - 1] * 4
            elif y == 0 and x > 0 and z == 0:
                self.local_sum[y, x, z] = \
                    self.image_constants.middle_sample_value * 4
        

    def __calculate_local_difference_vector(self, x, y, z, t):
        if t == 0:
            return

        offset = 0

        if self.header.prediction_mode == hd.PredictionMode.FULL:
            if x > 0 and y > 0:
                self.local_difference_vector[y,x,z,0] = 4 * self.sample_representative[y - 1, x    , z] - self.local_sum[y, x, z]
                self.local_difference_vector[y,x,z,1] = 4 * self.sample_representative[y    , x - 1, z] - self.local_sum[y, x, z]
                self.local_difference_vector[y,x,z,2] = 4 * self.sample_representative[y - 1, x - 1, z] - self.local_sum[y, x, z]
            elif x == 0 and y > 0:
                self.local_difference_vector[y,x,z,0] = 4 * self.sample_representative[y - 1, x    , z] - self.local_sum[y, x, z]
                self.local_difference_vector[y,x,z,1] = 4 * self.sample_representative[y - 1, x    , z] - self.local_sum[y, x, z]
                self.local_difference_vector[y,x,z,2] = 4 * self.sample_representative[y - 1, x    , z] - self.local_sum[y, x, z]
            else: # y = 0
                self.local_difference_vector[y,x,z,0] = 0
                self.local_difference_vector[y,x,z,1] = 0
                self.local_difference_vector[y,x,z,2] = 0
            offset += 3
        
        if z > 0:
            self.local_difference_vector[y,x,z,offset] = 4 * self.sample_representative[y, x, z - 1] - self.local_sum[y, x, z - 1]
            for i in range(1, self.spectral_bands_used[z]):
                self.local_difference_vector[y,x,z,offset + i] = self.local_difference_vector[y, x, z - 1, offset + i - 1]
        

    def __init_weights(self, z):
        if self.header.weight_init_method == hd.WeightInitMethod.DEFAULT:
            offset = 0
            if self.header.prediction_mode == hd.PredictionMode.FULL:
                self.weight_vector[0,1,z,0] = 0
                self.weight_vector[0,1,z,1] = 0
                self.weight_vector[0,1,z,2] = 0
                offset += 3

            if z > 0:
                self.weight_vector[0,1,z,offset] = 2**self.weight_component_resolution * 7 / 8
                for i in range(1, self.spectral_bands_used[z]):
                        self.weight_vector[0,1,z,offset + i] = self.weight_vector[0,1,z,offset + i - 1] / 8
        else:
            exit("Custom weight init method not supported")


    def __calculate_weight_vector(self, x, y, z, t):
        if t == 0:
            return
        if t == 1:
            self.__init_weights(z)
            return
        
        prev_y = y
        prev_x = x - 1
        if prev_x < 0:
            prev_y -= 1
            prev_x = self.header.x_size - 1
        
        assert t - 1 == prev_x + prev_y * self.header.x_size

        # TODO: Add weight exponent offset
        weight_exponent_offset = 0.0

        self.weight_vector[y,x,z] = \
            (self.weight_vector[prev_y,prev_x,z] + np.floor( \
            (float(sign_positive(self.double_resolution_prediction_error[prev_y,prev_x,z])) * \
            2.0**(-(self.weight_update_scaling_exponent[t - 1].astype(np.float64) + weight_exponent_offset)) * \
            self.local_difference_vector[prev_y,prev_x,z].astype(np.float64) + 1.0)/2.0).astype(np.int64) \
            ).clip(self.weight_min, self.weight_max)
        

    def __calculate_predicted_central_local_difference(self, x, y, z, t):
        if t == 0:
            return
        if self.header.prediction_mode == hd.PredictionMode.REDUCED and z == 0:
            self.predicted_central_local_difference[y,x,z] = 0
            return
        self.predicted_central_local_difference[y,x,z] = \
            np.dot(self.weight_vector[y,x,z], self.local_difference_vector[y,x,z])


    def __calculate_prediction(self, x, y, z, t):
        if t > 0:
            self.high_resolution_predicted_sample_value[y,x,z] = \
                clip( 
                    modulo_star(
                        self.predicted_central_local_difference[y,x,z] + \
                        2**self.weight_component_resolution * \
                        (self.local_sum[y,x,z] - 4 * self.image_constants.middle_sample_value), \
                        self.register_size \
                    ) + 2**(self.weight_component_resolution + 2) * \
                    self.image_constants.middle_sample_value + \
                    2**(self.weight_component_resolution + 1), \
                    2**(self.weight_component_resolution + 2) * self.image_constants.lower_sample_limit, \
                    2**(self.weight_component_resolution + 2) * self.image_constants.upper_sample_limit +
                    2**(self.weight_component_resolution + 1) \
                )
            self.double_resolution_predicted_sample_value[y,x,z] = \
                self.high_resolution_predicted_sample_value[y,x,z] / \
                2**(self.weight_component_resolution + 1)
        elif t == 0 and self.header.prediction_bands_num > 0 and z > 0:
            self.double_resolution_predicted_sample_value[y,x,z] = \
                2 * self.image_sample[y,x,z - 1]
        else:
            self.double_resolution_predicted_sample_value[y,x,z] = \
                2 * self.image_constants.middle_sample_value
            
        self.predicted_sample_value[y,x,z] = \
            self.double_resolution_predicted_sample_value[y,x,z] / 2
    

    def __calculate_maximum_error(self, x, y, z, t):
        # Assumes periodic_error_updating_flag = NOT_USED, absolute_error_limit_assignment_method = BAND_INDEPENDENT and relative_error_limit_assignment_method = BAND_INDEPENDENT
        if self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.LOSSLESS:
            self.maximum_error[y, x, z] = 0

        elif self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
            self.maximum_error[y, x, z] = self.header.absolute_error_limit_value

        elif self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.RELATIVE_ONLY:
            self.maximum_error[y, x, z] =  \
                int(self.header.relative_error_limit_value * \
                self.predicted_sample_value[y, x, z] / \
                self.image_constants.dynamic_range)
            
        else: # self.header.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE
            self.maximum_error[y, x, z] = \
                min( \
                    self.header.absolute_error_limit_value, \
                    int(self.header.relative_error_limit_value * \
                    self.predicted_sample_value[y, x, z] / \
                    self.image_constants.dynamic_range) \
                )

    
    def __calculate_quantization(self, x, y, z, t):
        self.prediction_residual[y,x,z] = \
            self.image_sample[y,x,z] - self.predicted_sample_value[y,x,z]

        if t == 0:
            self.quantizer_index[y,x,z] = self.prediction_residual[y,x,z]
            return
        
        self.quantizer_index[y,x,z] = \
            sign(self.prediction_residual[y,x,z]) * \
            int((abs(self.prediction_residual[y,x,z]) + \
            self.maximum_error[y,x,z]) / \
            (2 * self.maximum_error[y,x,z] + 1))
        
    
    def __calculate_sample_representative(self, x, y, z, t):
        if t == 0:
            self.sample_representative[y, x, z] = self.image_sample[y, x, z]
            return
        
        if self.maximum_error[y, x, z] == 0: # Lossless
            self.clipped_quantizer_bin_center[y, x, z] = self.image_sample[y, x, z]
        else:
            self.clipped_quantizer_bin_center[y, x, z] = \
                clip( \
                    self.predicted_sample_value[y, x, z] + \
                    self.quantizer_index[y, x, z] * \
                    (2 * self.maximum_error[y, x, z] + 1), \
                    self.image_constants.lower_sample_limit, \
                    self.image_constants.upper_sample_limit \
                )
        
        if self.header.fixed_damping_value == 0 and self.header.fixed_offset_value == 0: # Lossless
            self.double_resolution_sample_representative[y, x, z] = \
                2 * self.clipped_quantizer_bin_center[y, x, z]
            
            self.sample_representative[y, x, z] = self.image_sample[y, x, z]

        else:
            # Assumes band_varying_damping_flag = BAND_INDEPENDENT and band_varying_offset_flag = BAND_INDEPENDENT
            self.double_resolution_sample_representative[y, x, z] = \
                4 * (2**self.header.sample_representative_resolution - self.header.fixed_damping_value) * \
                (self.clipped_quantizer_bin_center[y, x, z] * 2**self.weight_component_resolution - \
                sign(self.quantizer_index[y, x, z]) * self.maximum_error[y, x, z] * \
                self.header.fixed_offset_value * \
                2**(self.weight_component_resolution - self.header.sample_representative_resolution)) + \
                self.header.fixed_damping_value * \
                self.high_resolution_predicted_sample_value[y, x, z] - \
                self.header.fixed_damping_value * 2**(self.weight_component_resolution + 1) / \
                2**(self.weight_component_resolution + self.header.sample_representative_resolution + 1)

            self.sample_representative[y, x, z] = \
                (self.double_resolution_sample_representative[y, x, z] + 1) / 2
            

    def __calculate_prediction_error(self, x, y, z, t):
        self.double_resolution_prediction_error[y, x, z] = \
            2 * self.clipped_quantizer_bin_center[y, x, z] - \
            self.double_resolution_predicted_sample_value[y, x, z]
        
    
    def __calculate_mapped_quantizer_index(self, x, y, z, t):
        if t == 0:
            self.scaled_prediction_endpoint_difference[y,x,z] = \
                min( \
                    self.predicted_sample_value[0, 0, z] - \
                    self.image_constants.lower_sample_limit, \
                    self.image_constants.upper_sample_limit - \
                    self.predicted_sample_value[0, 0, z] \
                )
        else:
            self.scaled_prediction_endpoint_difference[y,x,z] = \
                min( \
                    (self.predicted_sample_value[y, x, z] - \
                    self.image_constants.lower_sample_limit + \
                    self.maximum_error[y,x,z]) / \
                    (2 * self.maximum_error[y,x,z] + 1), \
                    (self.image_constants.upper_sample_limit - \
                    self.predicted_sample_value[y, x, z] + \
                    self.maximum_error[y,x,z]) / \
                    (2 * self.maximum_error[y,x,z] + 1) \
                )
        
        term = (-1)**(self.double_resolution_predicted_sample_value[y, x, z] % 2) * self.quantizer_index[y,x,z]

        if abs(self.quantizer_index[y,x,z]) > self.scaled_prediction_endpoint_difference[y,x,z]:
            self.mapped_quantizer_index[y, x, z] = \
                abs(self.quantizer_index[y,x,z]) + self.scaled_prediction_endpoint_difference[y,x,z]
        elif 0 <= term and term <= self.scaled_prediction_endpoint_difference[y,x,z]:
            self.mapped_quantizer_index[y, x, z] = \
                2 * abs(self.quantizer_index[y,x,z])
        else:
            self.mapped_quantizer_index[y, x, z] = \
                2 * abs(self.quantizer_index[y,x,z]) - 1


    def run_predictor(self):
        """Calculate the outputs of the predictor for the loaded image"""
        self.__init_predictor_constants()
        self.__init_predictor_arrays()

        for y in range(self.header.y_size):
            print(f"y={y}")
            for x in range(self.header.x_size):
                t = x + y * self.header.x_size
                for z in range(self.header.z_size):
                    self.__calculate_local_sum(x, y, z, t)
                    self.__calculate_local_difference_vector(x, y, z, t)
                    self.__calculate_weight_vector(x, y, z, t)
                    self.__calculate_predicted_central_local_difference(x, y, z, t)
                    self.__calculate_prediction(x, y, z, t)
                    self.__calculate_maximum_error(x, y, z, t)
                    self.__calculate_quantization(x, y, z, t)
                    self.__calculate_sample_representative(x, y, z, t)
                    self.__calculate_prediction_error(x, y, z, t)
                    self.__calculate_mapped_quantizer_index(x, y, z, t)

    def get_predictor_output(self):
        """Return the outputs of the predictor for the loaded image. The mapped quantizer index."""
        return self.mapped_quantizer_index    

    def save_data(self, output_folder):
        """Save the predictor data to csv files"""
        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        csv_vector_shape = (self.header.y_size * self.header.x_size, self.header.z_size * self.local_difference_values_num)
        np.savetxt(output_folder + "/predictor-00-local_sum.csv", self.local_sum.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-01-local_difference_vector.csv", self.local_difference_vector.reshape(csv_vector_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-02-weight_vector.csv", self.weight_vector.reshape(csv_vector_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-03-predicted_central_local_difference.csv", self.predicted_central_local_difference.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-04-high_resolution_predicted_sample_value.csv", self.high_resolution_predicted_sample_value.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-05-double_resolution_predicted_sample_value.csv", self.double_resolution_predicted_sample_value.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-06-predicted_sample_value.csv", self.predicted_sample_value.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-07-prediction_residual.csv", self.prediction_residual.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-08-maximum_error.csv", self.maximum_error.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-09-quantizer_index.csv", self.quantizer_index.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-10-clipped_quantizer_bin_center.csv", self.clipped_quantizer_bin_center.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-11-double_resolution_sample_representative.csv", self.double_resolution_sample_representative.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-12-sample_representative.csv", self.sample_representative.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-13-double_resolution_prediction_error.csv", self.double_resolution_prediction_error.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-14-mapped_quantizer_index.csv", self.mapped_quantizer_index.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-15-spectral_bands_used.csv", self.spectral_bands_used, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-16-image_sample.csv", self.image_sample.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-17-weight_update_scaling_exponent.csv", self.weight_update_scaling_exponent.reshape((self.header.y_size, self.header.x_size)), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/predictor-18-scaled_prediction_endpoint_difference.csv", self.scaled_prediction_endpoint_difference.reshape(csv_image_shape), delimiter=",", fmt='%d')
    