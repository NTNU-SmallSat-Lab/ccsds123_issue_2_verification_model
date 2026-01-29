#include "header_types.hpp"
#include "predictor.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>


#define DIMENSIONS (std::array<ssize_t, 3>){ _image_sample.shape(0), _image_sample.shape(1), _image_sample.shape(2) }
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
  _image_sample(image_sample.unchecked<3>()) // cast to three dimensional array
  
{
  init_predictor_constants();
  init_predictor_arrays();
}

/******************** Public ********************/

NumpyArr<int> Predictor::compress() {
  int x_size = header.attr("x_size").cast<int>();
  int y_size = header.attr("y_size").cast<int>();
  int z_size = header.attr("z_size").cast<int>();

  // mapped quantizer index
  auto  mqi = NumpyArr<int>(DIMENSIONS);
  auto _mqi = mqi.mutable_unchecked<3>();

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++) {
    std::cout << "\rProcessing line y=" << y+1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++) {
      int t = x + y * x_size;
      for (int z = 0; z < z_size; z++) {
        // local sum
        // local difference vector
        // weight vector
        // predicted central local difference
        // prediction
        // maximum error
        // quantization
        // sample representative
        // prediction error
        // mapped quantizer index
      }
    }
  }
  std::cout << std::endl;

  return mqi;
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
  lssmpl = new Sampler<int>(DIMENSIONS, true); // local sum sampler
}
