#include "predictor.hpp"
#include "header_types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#define SPECTRAL_BANDS_USED(z) std::min(z, header.attr("prediction_bands_num").cast<long>()) // symbol P^*
#define WEIGHT_UPDATE_SCALING_EXPONENT(t) std::clamp(weight_update_initial_parameter + x_size.cast<long>()) / weight_update_change_interval, \
                                                     weight_update_initial_parameter,                                                                            \
                                                     weight_update_final_parameter)                                                                              \
                                              + image_constants.attr("dynamic_range_bits").cast<long>()                                                          \
                                              - weight_component_resolution

/******************** Utils ********************/

// cast Python enum to C++ enum class
template <typename T>
static inline T cast_enum(py::object enum_py)
{
  auto value = enum_py.attr("value").cast<long>();
  return static_cast<T>(value);
}

static inline long modulo_star(long value, long r)
{
  if (r == 64)
    return value;

  long offset = 1 << (r - 1);
  long modulus = 1 << r;
  return ((value + offset) % modulus) - offset;
}

static inline long sgn(long value)
{
  if (value < 0)
    return -1;
  if (value > 0)
    return 1;
  return 0;
}

/******************** Constructor ********************/

Predictor::Predictor(py::object header, py::object image_constants, NumpyArr<long> image_sample, bool save_intermediates)
    : header(header),
      image_constants(image_constants),
      _image_sample(image_sample.mutable_unchecked<3>()), // cast to three dimensional array
      save_intermediates(save_intermediates)

{
  x_size = header.attr("x_size").cast<long>();
  y_size = header.attr("y_size").cast<long>();
  z_size = header.attr("z_size").cast<long>();

  init_predictor_constants();
  init_predictor_arrays();
}

/******************** Public ********************/

NumpyArr<long> Predictor::compress()
{
  // tranversing in BIP order
  for (int y = 0; y < y_size; y++)
  {
    std::cout << "\rProcessing line y=" << y + 1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++)
    {
      int t = x + y * x_size;
      long prev_local_sum; // local sum of previous band
      for (int z = 0; z < z_size; z++)
      {
        if (t == 0)
          continue;

        // local sum
        long local_sum = lssmpl->sample(calc_local_sum(y, x, z), x, y, z);

        // local difference vector
        auto local_difference_vector = calc_local_difference_vector(x, y, z, local_sum, prev_local_sum);
        for (int i = 0; i < local_difference_vector.size(); i++)
          ldvsmpl->sample(local_difference_vector.at(i), y, x, z, i);

        // predicted central local difference
        long predicted_central_local_diff = pcdsmpl->sample(calc_predicted_central_local_diff(x, y, z), x, y, z);

        // high resolution predicted sample value
        long high_resolution_pred_sample_value = hrpsvsmpl->sample(calc_high_resolution_pred_sample_value(x, y, z, local_sum, predicted_central_local_diff), x, y, z);

        // double resolution predicted sample value
        long double_resolution_predicted_sample_value = drpsvsmpl->sample(calc_double_resolution_predicted_sample_value(x, y, z, high_resolution_pred_sample_value), x, y, z);

        // predicted sample value
        long predicted_sample_value = psvsmpl->sample(calc_predicted_sample_value(double_resolution_predicted_sample_value), x, y, z);

        // prediction residual
        long prediction_residual = prsmpl->sample(calc_prediction_residual(_image_sample(y, x, z), predicted_sample_value), x, y, z);

        // max error value
        long maximum_error = mevsmpl->sample(calc_maximum_error(y, z, predicted_sample_value), x, y, z);

        // quantizer index
        long quantizer_index = qismpl->sample(calc_quantizer_index(t, maximum_error, prediction_residual), x, y, z);

        // clippped quantizer bin center
        long clipped_quantizer_bin_center = cqbcsmpl->sample(calc_clipped_quantizer_bin_center(x, y, z, predicted_sample_value, maximum_error, quantizer_index), x, y, z);

        // double resolution sample representative
        long double_resolution_sample_representative = drsrsmpl->sample(calc_double_resolution_sample_representative(z, clipped_quantizer_bin_center, quantizer_index, maximum_error, high_resolution_pred_sample_value), x, y, z);

        // sample representative
        long sample_representative = srsmpl->sample(calc_sample_representative(x, y, z, clipped_quantizer_bin_center, double_resolution_sample_representative), x, y, z);

        // double resolution prediction error
        long double_resolution_prediction_error = drpesmpl->sample(calc_double_resolution_prediction_error(clipped_quantizer_bin_center, double_resolution_predicted_sample_value), x, y, z);

        // weight update scaling exponent
        // weight update offset
        // weight update

        // theta
        long theta = tsmpl->sample(calc_theta(t, predicted_sample_value, maximum_error), x, y, z);

        // mapped quantizer index
        mqismpl->sample(calc_mapped_quantizer_index(quantizer_index, theta, double_resolution_predicted_sample_value), x, y, z);

        // for local difference calculation
        prev_local_sum = local_sum;
      }
    }
  }
  std::cout << std::endl;

  return mqismpl->get_arr();
}

void Predictor::save_data(std::string output_folder)
{
  // import numpy for easy storing to file
  py::object numpy = py::module_::import("numpy");
  py::object savetxt = numpy.attr("savetxt");

  auto csv_image_shape = {y_size * x_size, z_size};
  auto csv_vector_shape = {y_size * x_size, z_size * local_difference_values_num};

  // optionally saved
  if (lssmpl->enable_sampling)
    savetxt(output_folder + "/predictor-00-local_sum.csv", lssmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (pcdsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-03-predicted_central_local_difference.csv", pcdsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (hrpsvsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-04-high_resolution_predicted_sample_value.csv", hrpsvsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (drpsvsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-05-double_resolution_predicted_sample_value.csv", drpsvsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (psvsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-06-predicted_sample_value.csv", psvsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (prsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-07-prediction_residual.csv", prsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (qismpl->enable_sampling)
    savetxt(output_folder + "/predictor-09-quantizer_index.csv", qismpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (cqbcsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-10-clipper_quantizer_bin_center.csv", cqbcsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (drsrsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-11-double_resolution_sample_representative.csv", drsrsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (srsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-12-sample_representative.csv", srsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (tsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-18-scaled_prediction_endpoint_difference.csv", tsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (mevsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-22-maximum_error.csv", mevsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  // always saved
  savetxt(output_folder + "/predictor-01-local_difference_vector.csv", ldvsmpl->get_arr().reshape(csv_vector_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-02-weight_vectors.csv", wvsmpl->get_arr().reshape(csv_vector_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-12-sample_representative.csv", repsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-14-mapped_quantizer_index.csv", mqismpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-20-absolute_error_limits.csv", absolute_error_limits->get_arr(), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-21-relative_error_limits.csv", relative_error_limits->get_arr(), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
}

/******************** Private ********************/

void Predictor::init_predictor_constants()
{
  local_difference_values_num = header.attr("prediction_bands_num").cast<long>();
  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
    local_difference_values_num += 3;

  weight_component_resolution = header.attr("weight_component_resolution").cast<long>() + 4;
  weight_update_change_interval = (1 << header.attr("weight_update_change_interval").cast<long>()) + 4;
  weight_update_initial_parameter = header.attr("weight_update_initial_parameter").cast<long>() - 6;
  weight_update_final_parameter = header.attr("weight_update_final_parameter").cast<long>() - 6;

  register_size = header.attr("register_size").cast<long>();
  if (register_size == 0)
    register_size = 64;

  absolute_error_limits = new Sampler<long, 2>({y_size, z_size}, true);
  relative_error_limits = new Sampler<long, 2>({y_size, z_size}, true);

  QuantizerFidelityControlMethod fidelity_control = cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method"));
  bool periodic_error_updating_used = cast_enum<PeriodicErrorUpdatingFlag>(header.attr("periodic_error_updating_flag")) == PeriodicErrorUpdatingFlag::USED;

  // fill error limit arrays
  if (!periodic_error_updating_used && fidelity_control != QuantizerFidelityControlMethod::LOSSLESS)
  {
    auto relative_error_limit_table = header.attr("relative_error_limit_table").cast<NumpyArr<long>>().unchecked<1>();
    auto absolute_error_limit_table = header.attr("absolute_error_limit_table").cast<NumpyArr<long>>().unchecked<1>();

    if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
    {
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*absolute_error_limits)(y, z) = absolute_error_limit_table(z);
    }
    if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
    {
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*relative_error_limits)(y, z) = relative_error_limit_table(z);
    }
  }
  else if (periodic_error_updating_used)
  {
    auto periodic_relative_error_limit_table = header.attr("periodic_relative_error_limit_table").cast<NumpyArr<long>>().unchecked<2>();
    auto periodic_absolute_error_limit_table = header.attr("periodic_absolute_error_limit_table").cast<NumpyArr<long>>().unchecked<2>();

    long period = 1 << header.attr("error_update_period_exponent").cast<int>();
    for (int y = 0; y < (y_size + (1 << 16) * (int)(y_size == 0)); y++)
    {
      long i = y / period;
      if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
      {
        for (int y = 0; y < y_size; y++)
          for (int z = 0; z < z_size; z++)
            (*absolute_error_limits)(y, z) = periodic_absolute_error_limit_table(i, z);
      }
      if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
      {
        for (int y = 0; y < y_size; y++)
          for (int z = 0; z < z_size; z++)
            (*relative_error_limits)(y, z) = periodic_relative_error_limit_table(i, z);
      }
    }
  }
}

void Predictor::init_predictor_arrays()
{
  std::array<ssize_t, 3> image_shape = {_image_sample.shape(0), _image_sample.shape(1), _image_sample.shape(2)};
  std::array<ssize_t, 4> local_difference_vector_shape = {image_shape[0], image_shape[1], image_shape[2], local_difference_values_num};

  // these may be optionally not stored
  lssmpl = new Sampler<long, 3>(image_shape, save_intermediates);    // local sum
  pcdsmpl = new Sampler<long, 3>(image_shape, save_intermediates);   // predicted central local difference
  hrpsvsmpl = new Sampler<long, 3>(image_shape, save_intermediates); // high resolution predictied sample value
  drpsvsmpl = new Sampler<long, 3>(image_shape, save_intermediates); // double resolution predicted sample value
  psvsmpl = new Sampler<long, 3>(image_shape, save_intermediates);   // predicted sample value
  prsmpl = new Sampler<long, 3>(image_shape, save_intermediates);    // prediction residual
  mevsmpl = new Sampler<long, 3>(image_shape, save_intermediates);   // maximum error value
  qismpl = new Sampler<long, 3>(image_shape, save_intermediates);    // quantizer index
  cqbcsmpl = new Sampler<long, 3>(image_shape, save_intermediates);  // clipped quantizer bin center
  drsrsmpl = new Sampler<long, 3>(image_shape, save_intermediates);  // double resolution sample representative
  srsmpl = new Sampler<long, 3>(image_shape, save_intermediates);    // sample representative
  drpesmpl = new Sampler<long, 3>(image_shape, save_intermediates);  // double resolution prediction error
  tsmpl = new Sampler<long, 3>(image_shape, save_intermediates);     // scaled prediction endpoint difference (theta)
  mqismpl = new Sampler<long, 3>(image_shape, save_intermediates);   // mapped quantizer index

  // these must be stored as they are accessed during execution
  mqismpl = new Sampler<long, 3>(image_shape);                   // mapped quantizer indices
  repsmpl = new Sampler<long, 3>(image_shape);                   // sample representatives
  ldvsmpl = new Sampler<long, 4>(local_difference_vector_shape); // local difference vectors
  wvsmpl = new Sampler<long, 4>(local_difference_vector_shape);  // weight vectors
}

long Predictor::calc_local_sum(long x, long y, long z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local sum not defined for t=0");

  long local_sum;

  switch (cast_enum<LocalSumType>(header.attr("local_sum_type")))
  {

  case LocalSumType::WIDE_NEIGHBOR_ORIENTED:
    if (y > 0 && 0 < x && x < x_size - 1)
    {
      local_sum = (*repsmpl)(y, x - 1, z) + (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z);
    }
    else if (y == 0 && x > 0)
    {
      local_sum = (*repsmpl)(y, x - 1, z) * 4;
    }
    else if (y > 0 && x == 0)
    {
      local_sum = ((*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z)) * 2;
    }
    else if (y > 0 && x == x_size - 1)
    {
      local_sum = (*repsmpl)(y, x - 1, z) + (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) * 2;
    }
    break;

  case LocalSumType::NARROW_NEIGHBOR_ORIENTED:
    if (y > 0 && 0 < x && x < x_size - 1)
    {
      local_sum = (*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z) * 2 + (*repsmpl)(y - 1, x + 1, z);
    }
    else if (y == 0 && x > 0 && z > 0)
    {
      local_sum = (*repsmpl)(y, x - 1, z - 1) * 4;
    }
    else if (y > 0 && x == 0)
    {
      local_sum = ((*repsmpl)(y - 1, x, z) + (*repsmpl)(y - 1, x + 1, z)) * 2;
    }
    else if (y > 0 && x == x_size - 1)
    {
      local_sum = ((*repsmpl)(y - 1, x - 1, z) + (*repsmpl)(y - 1, x, z)) * 2;
    }
    else if (y == 0 && x > 0 && z == 0)
    {
      local_sum = image_constants.attr("middle_sample_value").cast<long>() * 4;
    }
    break;

  case LocalSumType::WIDE_COLUMN_ORIENTED:
    if (y > 0)
    {
      local_sum = (*repsmpl)(y - 1, x, z) * 4;
    }
    else if (y == 0 && x > 0)
    {
      local_sum = (*repsmpl)(y, x - 1, z) * 4;
    }
    break;

  case LocalSumType::NARROW_COLUMN_ORIENTED:
    if (y > 0)
    {
      local_sum = (*repsmpl)(y - 1, x, z) * 4;
    }
    else if (y == 0 && x > 0 && z > 0)
    {
      local_sum = (*repsmpl)(y, x - 1, z - 1) * 4;
    }
    else if (y == 0 && x > 0 && z == 0)
    {
      local_sum = image_constants.attr("middle_sample_value").cast<long>() * 4;
    }
    break;
  }
  return local_sum;
}

std::vector<long> Predictor::calc_local_difference_vector(long x, long y, long z, long local_sum, long prev_local_sum)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local difference vector not defined for t=0");

  std::vector<long> local_difference_vector;

  long offset = 0;

  if (cast_enum<PredictionMode>(header.attr("local_sum_type")) == PredictionMode::FULL)
  {
    for (int i = 0; i < 3; i++)
      local_difference_vector.push_back(0);

    if (x > 0 && y > 0)
    {
      local_difference_vector.at(0) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*repsmpl)(y, x - 1, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*repsmpl)(y - 1, x - 1, z) - local_sum;
    }
    else if (x == 0 && y > 0)
    {
      local_difference_vector.at(0) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*repsmpl)(y - 1, x, z) - local_sum;
    }

    offset += 3;
  }

  if (z > 0 && SPECTRAL_BANDS_USED(z) > 0)
    // local difference of z-1
    local_difference_vector.push_back(4 * (*repsmpl)(y, x, z - 1) - prev_local_sum);

  // copy local differences of previous vector
  for (int i = 1; i < SPECTRAL_BANDS_USED(z); i++)
    local_difference_vector.push_back((*ldvsmpl)(y, x, z - 1, offset + i - 1));

  return local_difference_vector;
}

long Predictor::calc_predicted_central_local_diff(long x, long y, long z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("predicted central local difference not defined for t=0");

  if (cast_enum<PredictionMode>(header.attr("local_sum_type")) == PredictionMode::REDUCED && z == 0)
    return 0;

  long acc = 0;
  for (int i = 0; i < local_difference_values_num; i++)
    acc += (*ldvsmpl)(y, x, z, i) * (*wvsmpl)(y, x, z, i);
  return acc;
}

long Predictor::calc_high_resolution_pred_sample_value(long x, long y, long z, long local_sum, long predicted_central_local_diff)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("high resolution predicted sample value not defined for t=0");

  long sMin = image_constants.attr("lower_sample_limit").cast<int>();
  long sMid = image_constants.attr("middle_sample_value").cast<int>();
  long sMax = image_constants.attr("upper_sample_limit").cast<int>();

  long high_resolution_pred_sample_vaue = modulo_star(
                                              (long)(predicted_central_local_diff + ((local_sum - (sMid << 2)) << weight_component_resolution)), register_size)
                                          + (sMid
                                             << (weight_component_resolution + 2))
                                          + (1 << (weight_component_resolution + 1));

  return std::clamp(high_resolution_pred_sample_vaue, sMin << (weight_component_resolution + 2), (sMax << (weight_component_resolution + 2)) + (1 << (weight_component_resolution + 1)));
}

long Predictor::calc_double_resolution_predicted_sample_value(long x, long y, long z, long high_resolution_pred_sample_value)
{
  long double_resolution_predicted_sample_value;

  if ((x == 0 && y == 0) && header.attr("prediction_bands_num").cast<long>() > 0 && z > 0)
  {
    double_resolution_predicted_sample_value = 2 * _image_sample(y, x, z - 1);
  }
  else if (x == 0 && y == 0)
  {
    double_resolution_predicted_sample_value = 2 * image_constants.attr("middle_sample_value").cast<int>();
  }
  else
  {
    double_resolution_predicted_sample_value = high_resolution_pred_sample_value >> (weight_component_resolution + 1);
  }

  return double_resolution_predicted_sample_value;
}

long Predictor::calc_predicted_sample_value(long double_resolution_predicted_sample_value)
{
  return double_resolution_predicted_sample_value >> 2;
}

long Predictor::calc_prediction_residual(long sample, long predicted_sample_value)
{
  return sample - predicted_sample_value;
}

long Predictor::calc_maximum_error(long y, long z, long predicted_sample_value)
{
  long maximum_error;

  switch (cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method")))
  {
  case QuantizerFidelityControlMethod::LOSSLESS:
    maximum_error = 0;
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_ONLY:
    maximum_error = (*absolute_error_limits)(y, z);
    break;

  case QuantizerFidelityControlMethod::RELATIVE_ONLY:
    maximum_error = (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<int>();
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_AND_RELATIVE:
    maximum_error = std::min((*absolute_error_limits)(y, z), (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<int>());
    break;
  }

  return maximum_error;
}

long Predictor::calc_quantizer_index(long t, long maximum_error, long prediction_residual)
{
  if (t == 0)
    return prediction_residual;
  else
    return sgn(prediction_residual) * (std::abs(prediction_residual) + maximum_error) / (2 * maximum_error + 1);
}

long Predictor::calc_clipped_quantizer_bin_center(long x, long y, long z, long predicted_sample_value, long maximum_error, long quantizer_index)
{
  long sMin = image_constants.attr("lower_sample_limit").cast<int>();
  long sMax = image_constants.attr("upper_sample_limit").cast<int>();

  if (maximum_error == 0)
    return _image_sample(y, x, z);

  return std::clamp(predicted_sample_value + quantizer_index * (2 * maximum_error + 1), sMin, sMax);
}

long Predictor::calc_double_resolution_sample_representative(long z, long clipped_quantizer_bin_center, long quantizer_index, long maximum_error, long high_resolution_pred_sample_value)
{
  auto damping_table_array = header.attr("damping_table_array").cast<NumpyArr<long>>().unchecked<1>();
  auto damping_offset_table_array = header.attr("damping_offset_table_array").cast<NumpyArr<long>>().unchecked<1>();

  long double_resolution_sample_representative;

  if (damping_table_array(0) == 0 && damping_offset_table_array(0) == 0)
  {
    double_resolution_sample_representative = 2 * clipped_quantizer_bin_center;
  }
  else
  {
    int sample_representative_resolution = header.attr("sample_representative_resolution").cast<int>();
    double_resolution_sample_representative = (4 * ((1 << sample_representative_resolution) - damping_table_array(z)) * (clipped_quantizer_bin_center * (1 << weight_component_resolution) - sgn(quantizer_index) * maximum_error * damping_offset_table_array(z) * (1 << (weight_component_resolution - sample_representative_resolution))) + damping_table_array(z) * high_resolution_pred_sample_value - damping_table_array(z) * (1 << (weight_component_resolution + 1))) / (1 << (weight_component_resolution + sample_representative_resolution + 1));
  }

  return double_resolution_sample_representative;
}

long Predictor::calc_sample_representative(long x, long y, long z, long clipped_quantizer_bin_center, long double_resolution_sample_representative)
{
  if (x == 0 && y == 0)
    return _image_sample(y, x, z);

  auto damping_table_array = header.attr("damping_table_array").cast<NumpyArr<long>>().unchecked<1>();
  auto damping_offset_table_array = header.attr("damping_offset_table_array").cast<NumpyArr<long>>().unchecked<1>();

  if (damping_table_array(0) == 0 && damping_offset_table_array(0) == 0)
    return clipped_quantizer_bin_center;

  return (double_resolution_sample_representative + 1) / 2;
}

long Predictor::calc_double_resolution_prediction_error(long clipped_quantizer_bin_center, long double_resolution_predicted_sample_value)
{
  return 2 * clipped_quantizer_bin_center - double_resolution_predicted_sample_value;
}

long Predictor::calc_theta(long t, long predicted_sample_value, long maximum_error)
{
  long sMin = image_constants.attr("lower_sample_limit").cast<int>();
  long sMax = image_constants.attr("upper_sample_limit").cast<int>();

  if (t == 0)
    return std::min(predicted_sample_value - sMin, sMax - predicted_sample_value);

  long denominator = 2 * maximum_error + 1;

  return std::min((predicted_sample_value - sMin + maximum_error) / denominator, (sMax - predicted_sample_value + maximum_error) / denominator);
}

long Predictor::calc_mapped_quantizer_index(long quantizer_index, long theta, long double_resolution_predicted_sample_value)
{
  long term = std::pow(-1, double_resolution_predicted_sample_value % 2) * quantizer_index;

  if (std::abs(quantizer_index) > theta)
  {
    return std::abs(quantizer_index) + theta;
  }
  else if (0 <= term && term <= theta)
  {
    return 2 * std::abs(quantizer_index);
  }
  else
  {
    return 2 * quantizer_index - 2;
  }
}
