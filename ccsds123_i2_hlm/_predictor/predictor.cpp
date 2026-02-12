#include "predictor.hpp"
#include "header_types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#define SPECTRAL_BANDS_USED(z) std::min(z, header.attr("prediction_bands_num").cast<long>()) // symbol P^*

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

static inline long sgn_positive(long value)
{
  if (value >= 0)
    return 1;
  return -1;
}

/******************** (de)Constructor ********************/

Predictor::Predictor(py::object header, py::object image_constants, NumpyArr<long> image_sample, bool save_intermediates)
    : header(header),
      image_constants(image_constants),
      _image_sample(image_sample.mutable_unchecked<3>()), // cast to three dimensional array
      save_intermediates(save_intermediates)
{
  x_size = header.attr("x_size").cast<long>();
  y_size = header.attr("y_size").cast<long>();
  z_size = header.attr("z_size").cast<long>();

#ifdef DEBUG
  save_intermediates = true;
#endif

  init_predictor_constants();
  init_predictor_arrays();
}

Predictor::~Predictor()
{
}

/******************** Public ********************/

NumpyArr<long> Predictor::compress()
{
  std::cout << "Compressing image..." << std::endl;

  long t_max = x_size * y_size - 1;

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++)
  {
    std::cout << "\rProcessing line y=" << y + 1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++)
    {
      int t = x + y * x_size;
      long prev_local_sum; // local sum of previous band for local difference calculation
      for (int z = 0; z < z_size; z++)
      {
        if (t == 0)
        {
          long double_resolution_predicted_sample_value = drpsvsmpl->sample(calc_double_resolution_predicted_sample_value(0, 0, z, 0), 0, 0, z);
          long predicted_sample_value = psvsmpl->sample(calc_predicted_sample_value(double_resolution_predicted_sample_value), 0, 0, z);

          long prediction_residual = prsmpl->sample(calc_prediction_residual(_image_sample(0, 0, z), predicted_sample_value), 0, 0, z);
          long quantizer_index = qismpl->sample(calc_quantizer_index(0, 0, prediction_residual), 0, 0, z);

          srsmpl->sample(calc_sample_representative(0, 0, z, 0, double_resolution_predicted_sample_value), 0, 0, z);

          long theta = tsmpl->sample(calc_theta(0, predicted_sample_value, 0), 0, 0, z);
          mqismpl->sample(calc_mapped_quantizer_index(quantizer_index, theta, double_resolution_predicted_sample_value), 0, 0, z);

          mevsmpl->sample(calc_maximum_error(0, z, predicted_sample_value), 0, 0, z);

          continue;
        }

        // local sum and difference
        long local_sum = lssmpl->sample(calc_local_sum(x, y, z), y, x, z);
        auto local_difference_vector = calc_local_difference_vector(x, y, z, local_sum, prev_local_sum);
        for (int i = 0; i < local_difference_vector.size(); i++)
          ldvsmpl->sample(local_difference_vector.at(i), y, x, z, i);
        long predicted_central_local_diff = pcdsmpl->sample(calc_predicted_central_local_diff(x, y, z), y, x, z);

        // prediction calculation
        long high_resolution_pred_sample_value = hrpsvsmpl->sample(calc_high_resolution_pred_sample_value(x, y, z, local_sum, predicted_central_local_diff), y, x, z);
        long double_resolution_predicted_sample_value = drpsvsmpl->sample(calc_double_resolution_predicted_sample_value(x, y, z, high_resolution_pred_sample_value), y, x, z);
        long predicted_sample_value = psvsmpl->sample(calc_predicted_sample_value(double_resolution_predicted_sample_value), y, x, z);

        // quantization
        long prediction_residual = prsmpl->sample(calc_prediction_residual(_image_sample(y, x, z), predicted_sample_value), y, x, z);
        long maximum_error = mevsmpl->sample(calc_maximum_error(y, z, predicted_sample_value), y, x, z);
        long quantizer_index = qismpl->sample(calc_quantizer_index(t, maximum_error, prediction_residual), y, x, z);
        long clipped_quantizer_bin_center = cqbcsmpl->sample(calc_clipped_quantizer_bin_center(x, y, z, predicted_sample_value, maximum_error, quantizer_index), y, x, z);

        // sample representatives
        long double_resolution_sample_representative = drsrsmpl->sample(calc_double_resolution_sample_representative(z, clipped_quantizer_bin_center, quantizer_index, maximum_error, high_resolution_pred_sample_value), y, x, z);
        srsmpl->sample(calc_sample_representative(x, y, z, clipped_quantizer_bin_center, double_resolution_sample_representative), y, x, z);
        long double_resolution_prediction_error = drpesmpl->sample(calc_double_resolution_prediction_error(clipped_quantizer_bin_center, double_resolution_predicted_sample_value), y, x, z);

        // mapping
        long theta = tsmpl->sample(calc_theta(t, predicted_sample_value, maximum_error), y, x, z);
        mqismpl->sample(calc_mapped_quantizer_index(quantizer_index, theta, double_resolution_predicted_sample_value), y, x, z);

        prev_local_sum = local_sum;

        if (t == t_max)
          continue;

        // weight update
        long x_weight_update = (t + 1) % x_size;
        long y_weight_update = (t + 1) / x_size;
        assert(x_weight_update < x_size && y_weight_update < y_size);

        auto updated_weight_vector = calc_weight_vector(x, y, z, double_resolution_prediction_error);
        for (int i = 0; i < updated_weight_vector.size(); i++) // sets weights for t+1
          wvsmpl->sample(updated_weight_vector.at(i), y_weight_update, x_weight_update, z, i);
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
    savetxt(output_folder + "/predictor-10-clipped_quantizer_bin_center.csv", cqbcsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (drsrsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-11-double_resolution_sample_representative.csv", drsrsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (drpesmpl->enable_sampling)
    savetxt(output_folder + "/predictor-13-double_resolution_prediction_error.csv", drpesmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (srsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-12-sample_representative.csv", srsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (tsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-18-scaled_prediction_endpoint_difference.csv", tsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  if (mevsmpl->enable_sampling)
    savetxt(output_folder + "/predictor-22-maximum_error.csv", mevsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");

  // always saved
  savetxt(output_folder + "/predictor-01-local_difference_vector.csv", ldvsmpl->get_arr().reshape(csv_vector_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-02-weight_vector.csv", wvsmpl->get_arr().reshape(csv_vector_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-12-sample_representative.csv", srsmpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-14-mapped_quantizer_index.csv", mqismpl->get_arr().reshape(csv_image_shape), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-20-absolute_error_limits.csv", absolute_error_limits->get_arr(), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
  savetxt(output_folder + "/predictor-21-relative_error_limits.csv", relative_error_limits->get_arr(), py::arg("delimiter") = ",", py::arg("fmt") = "%d");
}

/******************** Private ********************/

void Predictor::init_predictor_constants()
{
  local_difference_values_num = header.attr("prediction_bands_num").cast<long>();
  assert(local_difference_values_num < z_size && "Prediction bands (P) must be less than number of bands");

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
    local_difference_values_num += 3;

  weight_component_resolution = header.attr("weight_component_resolution").cast<long>() + 4;
  weight_update_change_interval = 1 << (header.attr("weight_update_change_interval").cast<long>() + 4);
  weight_update_initial_parameter = header.attr("weight_update_initial_parameter").cast<long>() - 6;
  weight_update_final_parameter = header.attr("weight_update_final_parameter").cast<long>() - 6;

  weight_exponent_offset = new Sampler<long, 2>({z_size, local_difference_values_num}, 0, true);

  // fill weight exponent offset array
  bool not_all_zero = cast_enum<WeightExponentOffsetFlag>(header.attr("weight_exponent_offset_flag")) == WeightExponentOffsetFlag::NOT_ALL_ZERO;
  if (local_difference_values_num > 0 && not_all_zero)
  {
    PredictionMode mode = cast_enum<PredictionMode>(header.attr("prediction_mode"));
    auto weight_exopnent_offset_table = header.attr("weight_exopnent_offset_table").cast<NumpyArr<long>>().unchecked<2>();

    if (mode == PredictionMode::FULL)
    {
      for (int z = 0; z < z_size; z++)
      {
        for (int i = 0; i < 3; i++)
          (*weight_exponent_offset)(z, i) = weight_exopnent_offset_table(z, 0);
        // TODO: complete this!
      }
    }
    else if (mode == PredictionMode::REDUCED)
    {
    }
  }

  weight_min = -(1 << (weight_component_resolution + 2));
  weight_max = (1 << (weight_component_resolution + 2)) - 1;

  register_size = header.attr("register_size").cast<long>();
  if (register_size == 0)
    register_size = 64;

  absolute_error_limits = new Sampler<long, 2>({y_size, z_size}, -1, true);
  relative_error_limits = new Sampler<long, 2>({y_size, z_size}, -1, true);

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
  lssmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);    // local sum
  pcdsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);   // predicted central local difference
  hrpsvsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates); // high resolution predictied sample value
  drpsvsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates); // double resolution predicted sample value
  psvsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);   // predicted sample value
  prsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);    // prediction residual
  mevsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);   // maximum error value
  qismpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);    // quantizer index
  cqbcsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);  // clipped quantizer bin center
  drsrsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);  // double resolution sample representative
  drpesmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);  // double resolution prediction error
  tsmpl = new Sampler<long, 3>(image_shape, -1, save_intermediates);     // scaled prediction endpoint difference (theta)

  // these must be stored as they are accessed during execution
  mqismpl = new Sampler<long, 3>(image_shape, -1);                  // mapped quantizer indices
  srsmpl = new Sampler<long, 3>(image_shape, -1);                   // sample representative
  ldvsmpl = new Sampler<long, 4>(local_difference_vector_shape, 0); // local difference vectors
  wvsmpl = new Sampler<long, 4>(local_difference_vector_shape, 0);  // weight vectors

  init_weights(); // populate the initial weight values

#ifdef DEBUG
  // sorted in order they are calculated
  lssmpl->set_reference("reference/predictor-00-local_sum.csv");                                   // 2 9 1
  ldvsmpl->set_reference("reference/predictor-01-local_difference_vector.csv");                    // 1 9 2 0
  pcdsmpl->set_reference("reference/predictor-03-predicted_central_local_difference.csv");         // 1 9 1
  hrpsvsmpl->set_reference("reference/predictor-04-high_resolution_predicted_sample_value.csv");   // 1 9 1
  drpsvsmpl->set_reference("reference/predictor-05-double_resolution_predicted_sample_value.csv"); // 1 9 1
  psvsmpl->set_reference("reference/predictor-06-predicted_sample_value.csv");                     // 1 9 1
  prsmpl->set_reference("reference/predictor-07-prediction_residual.csv");                         // 1 9 1
  mevsmpl->set_reference("reference/predictor-22-maximum_error.csv");                              // 1 9 3
  qismpl->set_reference("reference/predictor-09-quantizer_index.csv");                             // 1 9 1
  cqbcsmpl->set_reference("reference/predictor-10-clipped_quantizer_bin_center.csv");              // 1 9 2
  drsrsmpl->set_reference("reference/predictor-11-double_resolution_sample_representative.csv");   // 1 9 1
  srsmpl->set_reference("reference/predictor-12-sample_representative.csv");                       // 1 9 1
  drpesmpl->set_reference("reference/predictor-13-double_resolution_prediction_error.csv");        // 1 9 1
  tsmpl->set_reference("reference/predictor-18-scaled_prediction_endpoint_difference.csv");        // 1 9 1
  mqismpl->set_reference("reference/predictor-14-mapped_quantizer_index.csv");                     // 1 9 2
  wvsmpl->set_reference("reference/predictor-02-weight_vector.csv");                               // 1 9 1 0 - note that this one calculates for t+1
#endif                                                                                             // DEBUG
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
      local_sum = (*srsmpl)(y, x - 1, z) + (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z);
    else if (y == 0 && x > 0)
      local_sum = (*srsmpl)(y, x - 1, z) * 4;
    else if (y > 0 && x == 0)
      local_sum = ((*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z)) * 2;
    else if (y > 0 && x == x_size - 1)
      local_sum = (*srsmpl)(y, x - 1, z) + (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) * 2;
    break;

  case LocalSumType::NARROW_NEIGHBOR_ORIENTED:
    if (y > 0 && 0 < x && x < x_size - 1)
      local_sum = (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) * 2 + (*srsmpl)(y - 1, x + 1, z);
    else if (y == 0 && x > 0 && z > 0)
      local_sum = (*srsmpl)(y, x - 1, z - 1) * 4;
    else if (y > 0 && x == 0)
      local_sum = ((*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z)) * 2;
    else if (y > 0 && x == x_size - 1)
      local_sum = ((*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z)) * 2;
    else if (y == 0 && x > 0 && z == 0)
      local_sum = image_constants.attr("middle_sample_value").cast<long>() * 4;
    break;

  case LocalSumType::WIDE_COLUMN_ORIENTED:
    if (y > 0)
      local_sum = (*srsmpl)(y - 1, x, z) * 4;
    else if (y == 0 && x > 0)
      local_sum = (*srsmpl)(y, x - 1, z) * 4;
    break;

  case LocalSumType::NARROW_COLUMN_ORIENTED:
    if (y > 0)
      local_sum = (*srsmpl)(y - 1, x, z) * 4;
    else if (y == 0 && x > 0 && z > 0)
      local_sum = (*srsmpl)(y, x - 1, z - 1) * 4;
    else if (y == 0 && x > 0 && z == 0)
      local_sum = image_constants.attr("middle_sample_value").cast<long>() * 4;
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
      local_difference_vector.at(0) = 4 * (*srsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*srsmpl)(y, x - 1, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*srsmpl)(y - 1, x - 1, z) - local_sum;
    }
    else if (x == 0 && y > 0)
    {
      local_difference_vector.at(0) = 4 * (*srsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(1) = 4 * (*srsmpl)(y - 1, x, z) - local_sum;
      local_difference_vector.at(2) = 4 * (*srsmpl)(y - 1, x, z) - local_sum;
    }

    offset += 3;
  }

  if (z > 0 && SPECTRAL_BANDS_USED(z) > 0)
  {
    // local difference of z-1
    local_difference_vector.push_back(4 * (*srsmpl)(y, x, z - 1) - prev_local_sum);

    // copy local differences of previous vector
    for (int i = 1; i < SPECTRAL_BANDS_USED(z); i++)
      local_difference_vector.push_back((*ldvsmpl)(y, x, z - 1, offset + i - 1));
  }

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

  long sMin = image_constants.attr("lower_sample_limit").cast<long>();
  long sMid = image_constants.attr("middle_sample_value").cast<long>();
  long sMax = image_constants.attr("upper_sample_limit").cast<long>();

  long high_resolution_pred_sample_vaue = modulo_star(
                                              (predicted_central_local_diff + ((local_sum - (4 * sMid)) << weight_component_resolution)), register_size)
                                          + (sMid
                                             << (weight_component_resolution + 2))
                                          + (1 << (weight_component_resolution + 1));

  return std::clamp(high_resolution_pred_sample_vaue, sMin << (weight_component_resolution + 2), (sMax << (weight_component_resolution + 2)) + (1 << (weight_component_resolution + 1)));
}

long Predictor::calc_double_resolution_predicted_sample_value(long x, long y, long z, long high_resolution_pred_sample_value)
{
  long double_resolution_predicted_sample_value;
  long P = header.attr("prediction_bands_num").cast<long>();

  if ((x == 0 && y == 0) && P > 0 && z > 0)
    double_resolution_predicted_sample_value = 2 * _image_sample(y, x, z - 1);
  else if (x == 0 && y == 0 && (P == 0 || z == 0))
    double_resolution_predicted_sample_value = 2 * image_constants.attr("middle_sample_value").cast<int>();
  else if (x != 0 || y != 0)
    double_resolution_predicted_sample_value = high_resolution_pred_sample_value >> (weight_component_resolution + 1);

  return double_resolution_predicted_sample_value;
}

long Predictor::calc_predicted_sample_value(long double_resolution_predicted_sample_value)
{
  return double_resolution_predicted_sample_value >> 1;
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
    maximum_error = (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<long>();
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_AND_RELATIVE:
    maximum_error = std::min((*absolute_error_limits)(y, z), (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<long>());
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
  long sMin = image_constants.attr("lower_sample_limit").cast<long>();
  long sMax = image_constants.attr("upper_sample_limit").cast<long>();

  if (maximum_error == 0)
    return _image_sample(y, x, z);

  return std::clamp(predicted_sample_value + quantizer_index * (2 * maximum_error + 1), sMin, sMax);
}

long Predictor::calc_double_resolution_sample_representative(long z, long clipped_quantizer_bin_center, long quantizer_index, long maximum_error, long high_resolution_pred_sample_value)
{
  auto phi = header.attr("damping_table_array")
                 .cast<NumpyArr<long>>()
                 .unchecked<1>()(z);

  auto psi = header.attr("damping_offset_table_array")
                 .cast<NumpyArr<long>>()
                 .unchecked<1>()(z);

  long Theta = header.attr("sample_representative_resolution").cast<long>();
  long Omega = weight_component_resolution;

  // Shortcut from spec
  if (phi == 0 && psi == 0)
    return 2 * clipped_quantizer_bin_center;

  // Precompute powers of two
  long two_Theta = 1L << Theta;
  long two_Omega = 1L << Omega;
  long two_Omega_minus_Theta = 1L << (Omega - Theta);
  long two_Omega_plus_1 = 1L << (Omega + 1);
  long denom = 1L << (Omega + Theta + 1);

  // s′z(t) · 2^Ω
  long term_signal = clipped_quantizer_bin_center * two_Omega;

  // sgn(qz(t)) · mz(t) · ψz · 2^(Ω−Θ)
  long term_error =
      sgn(quantizer_index) * maximum_error * psi * two_Omega_minus_Theta;

  // 4 · (2^Θ − φz)
  long gain = 4 * (two_Theta - phi);

  // Full numerator of Eq. (47)
  long numerator =
      gain * (term_signal - term_error)
      + phi * high_resolution_pred_sample_value
      - phi * two_Omega_plus_1;

  // Divide by 2^(Ω+Θ+1)
  return numerator / denom;
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
  long sMin = image_constants.attr("lower_sample_limit").cast<long>();
  long sMax = image_constants.attr("upper_sample_limit").cast<long>();

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
    return 2 * std::abs(quantizer_index) - 1;
  }
}

void Predictor::init_weights()
{
  // t = 1
  int x = 1;
  int y = 0;

  if (cast_enum<WeightInitMethod>(header.attr("weight_init_method")) == WeightInitMethod::DEFAULT)
  {
    int offset = 0;

    if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
      offset += 3; // N, W and NW are already zero initialized

    for (long z = 1; z < z_size; z++)
    {
      if (SPECTRAL_BANDS_USED(z) == 0)
        continue;

      (*wvsmpl)(y, x, z, offset) = (1 << weight_component_resolution) * 7 / 8;

      for (int i = 1; i < SPECTRAL_BANDS_USED(z); i++)
        (*wvsmpl)(y, x, z, offset + i) = (*wvsmpl)(y, x, z, offset + i - 1) / 8;
    }
  }
  else
  {
    // TODO: custom weight init
  }
}

long calc_weight_offset(long local_diff, long weight_update_scaling_exponent, long weight_exponent_offset, long double_resolution_prediction_error)
{
  long exponent = weight_update_scaling_exponent + weight_exponent_offset;
  if (exponent > 0)
    return ((((sgn_positive(double_resolution_prediction_error) * local_diff) >> exponent) + 1) >> 1);
  else
    return ((((sgn_positive(double_resolution_prediction_error) * local_diff) << (-exponent)) + 1) >> 1);
}

std::vector<long> Predictor::calc_weight_vector(long x, long y, long z, long double_resolution_prediction_error)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("weight vector not defined for t=0");

  long t = x + y * x_size;
  long weight_update_scaling_exponent = std::clamp(weight_update_initial_parameter + (t - x_size) / weight_update_change_interval,
                                                   weight_update_initial_parameter,
                                                   weight_update_final_parameter)
                                        + image_constants.attr("dynamic_range_bits").cast<long>()
                                        - weight_component_resolution;

  std::vector<long> weight_vector(local_difference_values_num);

  // calculates weight vector for t+1
  for (int i = 0; i < weight_vector.size(); i++)
  {
    long local_diff = (*ldvsmpl)(y, x, z, i);
    long weo = (*weight_exponent_offset)(z, i);

    long weight_offset = calc_weight_offset(local_diff, weight_update_scaling_exponent, weo, double_resolution_prediction_error);
    long weight_unclipped = (*wvsmpl)(y, x, z, i) + weight_offset;

    weight_vector.at(i) = std::clamp(weight_unclipped, weight_min, weight_max);
  }

  return weight_vector;
}
