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
  // tranversing in BIP order
  for (int y = 0; y < y_size; y++) {
    std::cout << "\rProcessing line y=" << y+1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++) {
      int t = x + y * x_size;
      for (int z = 0; z < z_size; z++) {
        if (t == 0) continue;

        // local sum
        int local_sum = lssmpl->sample(calc_local_sum(y, x, z, repsmpl), x, y, z);

        // local difference vector
        auto local_difference_vector = calc_local_difference_vector(x, y, z, local_sum, repsmpl, ldvsmpl);
        for (int i = 0; i < local_difference_vector.size(); i++)
          ldvsmpl->sample(local_difference_vector.at(i), y, x, z, i);

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

  return mqismpl->get_arr();
}

void Predictor::save_data(std::string output_folder) {
  // import numpy for easy storing to file
  py::object numpy = py::module_::import("numpy");
  py::object savetxt = numpy.attr("savetxt");

  auto csv_image_shape = { y_size * x_size, z_size };
  auto csv_vector_shape = { y_size * x_size, z_size * local_difference_values_num };

  if (lssmpl->enable_sampling)
    savetxt(output_folder + "/predictor-00-local_sum.csv", lssmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter")=",", py::arg("fmt")="%d");

  if (ldvsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-01-local_difference_vector.csv", ldvsmpl->get_arr().reshape(csv_vector_shape), py::arg("delimiter")=",", py::arg("fmt")="%d");
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

  // these may be optionally not stored
  lssmpl = new Sampler<int, 3>(image_shape, true); // local sum

  // these must be stored as they are accessed during execution
  mqismpl = new Sampler<int, 3>(image_shape); // mapped quantizer indices
  repsmpl = new Sampler<int, 3>(image_shape); // sample representatives
  ldvsmpl = new Sampler<int, 4>(local_difference_vector_shape); // local difference vectors
}

int Predictor::calc_local_sum(int x, int y, int z, Sampler<int, 3> *repsmpl) {
  if (x == 0 && y == 0) 
    throw std::invalid_argument("local sum not defined for t=0");

  int local_sum;

  switch (cast_enum<LocalSumType>(header.attr("local_sum_type"))) {

    case LocalSumType::WIDE_NEIGHBOR_ORIENTED:
      if (y > 0 && 0 < x && x < x_size - 1) { local_sum = (*repsmpl)(y, x - 1, z) + (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z); }
      else if (y == 0 && x > 0) { local_sum = (*repsmpl)(y, x - 1, z) * 4; }
      else if (y > 0 && x == 0) { local_sum = ((*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z)) * 2; }
      else if (y > 0 && x == x_size - 1) { local_sum = (*repsmpl)(y, x - 1, z) + (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) * 2; }
      break;

    case LocalSumType::NARROW_NEIGHBOR_ORIENTED:
      if (y > 0 && 0 < x && x < x_size - 1) { local_sum = (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) * 2 + (*repsmpl)(y - 1, x + 1, z); }
      else if (y == 0 && x > 0 && z > 0) { local_sum = (*repsmpl)(y, x - 1, z - 1) * 4; }
      else if (y > 0 && x == 0) { local_sum = ((*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z)) * 2; }
      else if (y > 0 && x == x_size - 1) { local_sum = ((*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z)) * 2; }
      else if (y == 0 && x > 0 && z == 0) { local_sum = image_constants.attr("middle_sample_value").cast<int>() * 4; }
      break;

    case LocalSumType::WIDE_COLUMN_ORIENTED:
      if (y > 0) { local_sum = (*repsmpl)(y - 1, x, z) * 4; }
      else if (y == 0 && x > 0) { local_sum = (*repsmpl)(y, x - 1, z) * 4; }
      break;
      
    case LocalSumType::NARROW_COLUMN_ORIENTED:
      if (y > 0) { local_sum = (*repsmpl)(y - 1, x, z) * 4; }
      else if (y == 0 && x > 0 && z > 0) { local_sum = (*repsmpl)(y, x - 1, z - 1) * 4; }
      else if (y == 0 && x > 0 && z == 0) { local_sum = image_constants.attr("middle_sample_value").cast<int>() * 4; }
      break;
  }
  return local_sum;
}

std::vector<int> Predictor::calc_local_difference_vector(int x, int y, int z, int local_sum, Sampler<int, 3> *repsmpl, Sampler<int, 4> *ldvsmpl) {
  if (x == 0 && y == 0) 
    throw std::invalid_argument("local sum not defined for t=0");

  std::vector<int> local_difference_vector;

  int offset = 0;

  if (cast_enum<PredictionMode>(header.attr("local_sum_type")) == PredictionMode::FULL) {
    for (int i = 0; i < 3; i++)
      local_difference_vector.push_back(0);

    if (x > 0 && y > 0) {
      local_difference_vector.at(0) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*repsmpl)(y, x - 1, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*repsmpl)(y - 1, x - 1, z) - local_sum;
    } else if (x == 0 && y > 0) {
      local_difference_vector.at(0) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
    }

    offset += 3;
  }

  // copies the local difference value of the previous vector
  if (z > 0 && SPECTRAL_BANDS_USED(z) > 0)
    for (int i = 0; i < SPECTRAL_BANDS_USED(z); i++)
      local_difference_vector.push_back((*ldvsmpl)(y, x, z - 1, offset + i - 1));

  return local_difference_vector;
}
