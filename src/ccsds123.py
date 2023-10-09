from . import header as hd
import numpy as np

def clip(x, min, max):
    if x < min:
        return min
    if x > max:
        return max
    return x

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

# def sign_positive(x):
#     if x >= 0:
#         return 1
#     return -1


class CCSDS123():
    """
    CCSDS 123.0-B-2 high level model class
    """
    header = None
    raw_image_folder = "raw_images"
    image_sample = None # Symbol: s
    output_folder = "output"

    def __init__(self, image_name):
        self.image_name = image_name
        self.header = hd.Header(image_name)

    def load_raw_image(self):
        """Load a raw image into a N_x * N_y by N_z array"""
        # Doing it the obvious way skipped the first byte, hence some hoops have been jumped through to fix it
        file = open(self.raw_image_folder + "/" + self.image_name, 'rb').read()
        first = file[0] # Set aside first byte
        last = (file[-2] << 8) + file[-1] # set aside last two bytes
        with open(self.raw_image_folder + "/" + self.image_name, 'rb') as file:
            file.seek(1) # Skip the first byte
            self.image_sample = np.fromfile(file, dtype=np.uint16)
        if self.image_sample.shape[0] != self.header.z_size * self.header.y_size * self.header.x_size:
            self.image_sample=np.pad(self.image_sample, (0,1), 'constant', constant_values=last) # Pad the last two bytes
        self.image_sample[0] += first << 8 # Reintroduce the first byte
        # self.image_raw.shape = (self.header.z_size, self.image_raw.size//self.header.z_size)
        self.image_sample = self.image_sample.reshape((self.header.z_size, self.header.y_size, self.header.x_size)) # Reshape to z,y,x (BSQ) 3D array
        self.image_sample = self.image_sample.transpose(1,2,0) # Transpose to y,x,z order (BIP) 

    # Image constants
    dynamic_range_bits = None # Symbol: D
    dynamic_range = None # 2^D
    lower_sample_limit = None # Symbol: s_min
    upper_sample_limit = None # Symbol: s_max
    middle_sample_value = None # Symbol: s_mid

    def __init_image_constants(self):
        self.dynamic_range_bits = self.header.dynamic_range
        if self.dynamic_range_bits == 0:
            self.dynamic_range_bits = 16
        if self.header.large_d_flag == hd.LargeDFlag.LARGE_D:
            self.dynamic_range_bits += 16

        self.dynamic_range = 2**self.dynamic_range_bits

        if self.header.sample_type == hd.SampleType.UNSIGNED_INTEGER:
            self.lower_sample_limit = 0
            self.upper_sample_limit = 2 ** self.dynamic_range_bits - 1
            self.middle_sample_value = 2 ** (self.dynamic_range_bits - 1)
        elif self.header.sample_type == hd.SampleType.SIGNED_INTEGER:
            self.lower_sample_limit = -2 ** (self.dynamic_range_bits - 1)
            self.upper_sample_limit = 2 ** (self.dynamic_range_bits - 1) - 1
            self.middle_sample_value = 0

    # Predictor constants
    local_difference_values_num = None

    spectral_bands_used = None # Symbol: P^*. Indexed by z
    spectral_bands_used_mask = None

    weight_component_resolution = None # Symbol: Omega
    weight_update_change_interval = None # Symbol: t_inc
    weight_update_initial_parameter = None # Symbol: nu_min
    weight_update_final_parameter = None # Symbol: nu_max
    weight_update_scaling_exponent = None # Symbol: rho. Indexed by t
    weight_min = None # Symbol: omega_min
    weight_max = None # Symbol: omega_max

    sample_representative_part_1 = None # 4 * (2^Theta - phi)
    sample_representative_part_2 = None # 2^Omega
    sample_representative_part_3 = None # psi * 2^(Omega - Theta)
    sample_representative_part_4 = None # phi * 2^(Omega + 1)
    sample_representative_part_5 = None # 2^(Omega + Theta + 1)


    def __init_predictor_constants(self):

        self.local_difference_values_num = self.header.prediction_bands_num
        if self.header.prediction_mode == hd.PredictionMode.FULL:
            self.local_difference_values_num += 3

        self.spectral_bands_used = np.empty((self.header.z_size), dtype=np.int32)
        for z in range(self.header.z_size):
            self.spectral_bands_used[z] = min(z, self.header.prediction_bands_num)

        self.spectral_bands_used_mask = np.empty((self.header.z_size, self.local_difference_values_num), dtype=np.int32)
        for z in range(self.header.z_size):
            offset = 0
            if self.header.prediction_mode == hd.PredictionMode.FULL:
                self.spectral_bands_used_mask[z, 0:3] = [1, 1, 1]
                offset = 3
            for i in range(self.header.prediction_bands_num):
                self.spectral_bands_used_mask[z, offset + i] = int(i < self.spectral_bands_used[z])

        self.weight_component_resolution = self.header.weight_component_resolution + 4
        self.weight_update_change_interval = 2**self.header.weight_update_change_interval
        self.weight_update_initial_parameter = self.header.weight_update_initial_parameter - 6
        self.weight_update_final_parameter = self.header.weight_update_final_parameter - 6
        self.weight_update_scaling_exponent = np.empty((self.header.y_size * self.header.x_size), dtype=np.int32)
        for t in range (self.header.y_size * self.header.x_size):
            self.weight_update_scaling_exponent[t] = clip( \
                self.weight_update_initial_parameter + int((t - self.header.x_size) / self.weight_update_change_interval) \
                , self.weight_update_initial_parameter, self.weight_update_final_parameter) \
                + self.dynamic_range_bits - self.weight_component_resolution
        self.weight_min = -2**(self.weight_component_resolution + 2)
        self.weight_max = 2**(self.weight_component_resolution + 2) - 1

        self.sample_representative_part_1 = 4 * (2**self.header.sample_representative_resolution - self.header.fixed_damping_value)
        self.sample_representative_part_2 = 2**self.weight_component_resolution
        self.sample_representative_part_3 = self.header.fixed_offset_value * 2**(self.weight_component_resolution - self.header.sample_representative_resolution)
        self.sample_representative_part_4 = self.header.fixed_damping_value * 2**(self.weight_component_resolution + 1)
        self.sample_representative_part_5 = 2**(self.weight_component_resolution + self.header.sample_representative_resolution + 1)


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
    mapped_quantizer_index = None # Symbol: delta
    
    def __init_predictor_arrays(self):
        image_shape = self.image_sample.shape
        local_difference_vector_shape = image_shape + (self.local_difference_values_num,) 

        value = -1

        self.local_sum = np.full(image_shape, value, dtype=np.int32)
        self.local_difference_vector = np.full(local_difference_vector_shape, value, dtype=np.int32)
        self.weight_vector = np.full(local_difference_vector_shape, value, dtype=np.int32)
        self.predicted_central_local_difference = np.full(image_shape, value, dtype=np.int32)
        self.high_resolution_predicted_sample_value = np.full(image_shape, value, dtype=np.int32)
        self.double_resolution_predicted_sample_value = np.full(image_shape, value, dtype=np.int32)
        self.predicted_sample_value = np.full(image_shape, value, dtype=np.int32)
        self.prediction_residual = np.full(image_shape, value, dtype=np.int32)
        self.maximum_error = np.full(image_shape, value, dtype=np.int32)
        self.quantizer_index = np.full(image_shape, value, dtype=np.int32)
        self.clipped_quantizer_bin_center = np.full(image_shape, value, dtype=np.int32)
        self.double_resolution_sample_representative = np.full(image_shape, value, dtype=np.int32)
        self.sample_representative = np.full(image_shape, value, dtype=np.int32)
        self.double_resolution_prediction_error = np.full(image_shape, value, dtype=np.int32)
        self.mapped_quantizer_index = np.full(image_shape, value, dtype=np.int32)

        # See standard 4.7.3. TODO: Move elsewhere?
        self.double_resolution_predicted_sample_value[0,0,0] = 2 * self.middle_sample_value
        self.predicted_sample_value[0,0,0] = self.middle_sample_value


    def __calculate_maximum_error(self, x, y, z, t):
        # Assumes periodic_error_updating_flag = NOT_USED, absolute_error_limit_assignment_method = BAND_INDEPENDENT and relative_error_limit_assignment_method = BAND_INDEPENDENT
        if self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.LOSSLESS:
            self.maximum_error[y, x, z] = 0
        elif self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
            self.maximum_error[y, x, z] = self.header.absolute_error_limit_value
        elif self.header.quantizer_fidelity_control_method == hd.QuantizerFidelityControlMethod.RELATIVE_ONLY:
            self.maximum_error[y, x, z] = int(self.header.relative_error_limit_value * self.predicted_sample_value[y, x, z] / self.dynamic_range)
        else: # self.header.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE
            self.maximum_error[y, x, z] = min(self.header.absolute_error_limit_value, int(self.header.relative_error_limit_value * self.predicted_sample_value[y, x, z] / self.dynamic_range))
        

    def __calculate_sample_representative(self, x, y, z, t):
        if t == 0:
            self.sample_representative[y, x, z] = self.image_sample[y, x, z]
            return
        
        if self.maximum_error[y, x, z] == 0: # Lossless
            self.clipped_quantizer_bin_center[y, x, z] = self.image_sample[y, x, z]
        else:
            self.clipped_quantizer_bin_center[y, x, z] = clip(self.predicted_sample_value[y, x, z] + self.quantizer_index[y, x, z] * (2 * self.maximum_error[y, x, z] + 1), self.lower_sample_limit, self.upper_sample_limit)
        
        if self.header.fixed_damping_value == 0 and self.header.fixed_offset_value == 0: # Lossless
            self.double_resolution_sample_representative[y, x, z] = self.clipped_quantizer_bin_center[y, x, z]
        else:
            # Assumes band_varying_damping_flag = BAND_INDEPENDENT and band_varying_offset_flag = BAND_INDEPENDENT
            self.double_resolution_sample_representative[y, x, z] = \
                int((self.sample_representative_part_1 * \
                (self.clipped_quantizer_bin_center[y, x, z] * self.sample_representative_part_2 - \
                sign(self.quantizer_index[y, x, z]) * self.maximum_error[y, x, z] * \
                self.sample_representative_part_3) + \
                self.header.fixed_damping_value * self.high_resolution_predicted_sample_value[y, x, z] \
                - self.sample_representative_part_4) \
                / self.sample_representative_part_5)
            
        self.sample_representative[y, x, z] = int(self.double_resolution_sample_representative[y, x, z] / 2)

    
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
                    self.middle_sample_value * 4

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
                    self.middle_sample_value * 4
        

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

        # TODO: Add weight exponent offset
        weight_exponent_offset = 0

        double_resolution_prediction_error_sign_positive = \
            np.sign(self.double_resolution_prediction_error[prev_y,prev_x,z]) + \
            (self.double_resolution_prediction_error[prev_y,prev_x,z] == 0).astype(int)
        
        self.weight_vector[y,x,z] = \
            self.spectral_bands_used_mask[z] * \
            (self.weight_vector[prev_y,prev_x,z] + \
            1/2 * (double_resolution_prediction_error_sign_positive * \
            2**(-(self.weight_update_scaling_exponent[t - 1] + weight_exponent_offset)) * \
            self.local_difference_vector[prev_y,prev_x,z] + 1) \
            ).clip(self.weight_min, self.weight_max)
                

    def predictor(self):
        """Calculate the outputs of the predictor for the loaded image"""
        self.__init_image_constants()
        self.__init_predictor_constants()
        self.__init_predictor_arrays()

        for y in range(self.header.y_size):
            print(f"y={y}")
            for x in range(self.header.x_size):
                t = x + y * self.header.x_size
                for z in range(self.header.z_size):
                    self.__calculate_maximum_error(x, y, z, t)
                    self.__calculate_sample_representative(x, y, z, t)
                    self.__calculate_local_sum(x, y, z, t)
                    self.__calculate_local_difference_vector(x, y, z, t)
                    self.__calculate_weight_vector(x, y, z, t)


    def save_data(self):
        np.savetxt(self.output_folder + "/" + "00-local_sum.csv", self.local_sum.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "01-local_difference_vector.csv", self.local_difference_vector.reshape((self.header.y_size * self.header.x_size, self.header.z_size * self.local_difference_values_num)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "02-weight_vector.csv", self.weight_vector.reshape((self.header.y_size * self.header.x_size, self.header.z_size * self.local_difference_values_num)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "03-predicted_central_local_difference.csv", self.predicted_central_local_difference.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "04-high_resolution_predicted_sample_value.csv", self.high_resolution_predicted_sample_value.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "05-double_resolution_predicted_sample_value.csv", self.double_resolution_predicted_sample_value.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "06-predicted_sample_value.csv", self.predicted_sample_value.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "07-prediction_residual.csv", self.prediction_residual.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "08-maximum_error.csv", self.maximum_error.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "09-quantizer_index.csv", self.quantizer_index.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "10-clipped_quantizer_bin_center.csv", self.clipped_quantizer_bin_center.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "11-double_resolution_sample_representative.csv", self.double_resolution_sample_representative.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "12-sample_representative.csv", self.sample_representative.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "13-double_resolution_prediction_error.csv", self.double_resolution_prediction_error.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "14-mapped_quantizer_index.csv", self.mapped_quantizer_index.reshape((self.header.y_size * self.header.x_size, self.header.z_size)), delimiter=",", fmt='%d')
        np.savetxt(self.output_folder + "/" + "15-spectral_bands_used.csv", self.spectral_bands_used, delimiter=",", fmt='%d')



