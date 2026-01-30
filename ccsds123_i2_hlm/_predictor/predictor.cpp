#include "header_types.hpp"
#include "predictor.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>

 
#define SPECTRAL_BANDS_USED(z) std::min(z, header.attr("prediction_bands_num").cast<int>()) // symbol P^*
#define WEIGHT_UPDATE_SCALING_EXPONENT(t) std::clamp(weight_update_initial_parameter +                         \
                                                     (t - header.attr("x_size").cast<int>())                   \
                                                     / weight_update_change_interval,                          \
                                                     weight_update_initial_parameter,                          \
                                                     weight_update_final_parameter                             \
                                                    ) + image_constants.attr("dynamic_range_bits").cast<int>() \
                                                    - weight_component_resolution


// cast Python enum to C++ enum class
template <typename T>
T cast_enum(py::object enum_py) {
  auto value = enum_py.attr("value").cast<int>();
  return static_cast<T>(value);
}

/******************** Constructor ********************/

Predictor::Predictor(py::object header, py::object image_constants, NumpyArr<int> image_sample) : 
  header(header),
  image_constants(image_constants),
  _image_sample(image_sample.mutable_unchecked<3>()) // cast to three dimensional array
  
{
  x_size = header.attr("x_size").cast<int>();
  y_size = header.attr("y_size").cast<int>();
  z_size = header.attr("z_size").cast<int>();

  init_predictor_constants();
  init_predictor_arrays();
}

/******************** Public ********************/

NumpyArr<int> Predictor::compress() {
  std::array<ssize_t, 3> image_shape = { _image_sample.shape(0), _image_sample.shape(1), _image_sample.shape(2) };

  // mapped quantizer index
  auto  mqi = NumpyArr<int>(image_shape);
  auto _mqi = mqi.mutable_unchecked<3>();

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++) {
    std::cout << "\rProcessing line y=" << y+1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++) {
      int t = x + y * x_size;
      for (int z = 0; z < z_size; z++) {
        // local sum
        lssmpl->sample(x + y + z, x, y, z);
        // predicted central difference
        // high resolution predicted sample value
        // double resolution predicted sample value
        // predicted sample value
        // prediction residual
        // max error value
        // quatizer index
        // clippped quantizer bin center
        // double resolution sample representative
        // double resolution prediction error
        // weight update scaling exponent ?
        // weight update offset
        // weight update
        // theta
        // mapped quantizer index
        // sample representative
      }
    }
  }
  std::cout << std::endl;

  return mqi;
}

void Predictor::save_data(std::string output_folder) {
  // import numpy for easy storing to file
  py::object numpy = py::module_::import("numpy");
  py::object savetxt = numpy.attr("savetxt");

  auto csv_image_shape = { y_size * x_size, z_size };

  if (lssmpl->enable_sampling)
    savetxt(output_folder + "/predictor-00-local_sum.csv", lssmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter")=",", py::arg("fmt")="%d");
}

/******************** Private ********************/

void Predictor::init_predictor_constants() {
  local_difference_values_num = header.attr("prediction_bands_num").cast<int>();
  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
    local_difference_values_num += 3;

  weight_component_resolution = header.attr("weight_component_resolution").cast<int>() + 4;
  weight_update_change_interval = std::pow(2, header.attr("weight_update_change_interval").cast<int>() + 4);
  weight_update_initial_parameter = header.attr("weight_update_initial_parameter").cast<int>() - 6;
  weight_update_final_parameter = header.attr("weight_update_final_parameter").cast<int>() - 6;
}

void Predictor::init_predictor_arrays() {
  std::array<ssize_t, 3> image_shape = { _image_sample.shape(0), _image_sample.shape(1), _image_sample.shape(2) };
  std::array<ssize_t, 4> local_difference_vector_shape = { image_shape[0], image_shape[1], image_shape[2], local_difference_values_num };

  lssmpl = new Sampler<int, 3>(image_shape, true); // local sum sampler
  ldvsmpl = new Sampler<int, 4>(local_difference_vector_shape, true); // local difference vector sampler
}
