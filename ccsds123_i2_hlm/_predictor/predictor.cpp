#include "predictor.hpp"
#include "header_types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#define SPECTRAL_BANDS_USED(z) std::min(z, header.attr("prediction_bands_num").cast<long long>()) // symbol P^*

/******************** Utils ********************/

// cast Python enum to C++ enum class
template <typename T>
static inline T cast_enum(py::object enum_py)
{
  auto value = enum_py.attr("value").cast<long long>();
  return static_cast<T>(value);
}

static inline long long modulo_star(long long value, long long r)
{
  if (r == 64)
    return value;

  long long offset = 1LL << (r - 1);
  long long modulus = 1LL << r;
  return ((value + offset) % modulus) - offset;
}

static inline long long sgn(long long value)
{
  if (value < 0)
    return -1;
  if (value > 0)
    return 1;
  return 0;
}

static inline long long sgn_positive(long long value)
{
  if (value >= 0)
    return 1;
  return -1;
}

/******************** (de)Constructor ********************/

Predictor::Predictor(py::object header, py::object image_constants, NumpyArr<long long> image_sample, bool save_intermediates)
    : header(header),
      image_constants(image_constants),
      image_sample(image_sample),
      save_intermediates(save_intermediates)
{
  x_size = header.attr("x_size").cast<long long>();
  y_size = header.attr("y_size").cast<long long>();
  z_size = header.attr("z_size").cast<long long>();

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

NumpyArr<long long> Predictor::compress()
{
  std::cout << "Compressing image..." << std::endl;

  long long t_max = x_size * y_size - 1;
  auto _image_sample = image_sample.unchecked<3>();

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++)
  {
    std::cout << "\rProcessing line y=" << y + 1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++)
    {
      int t = x + y * x_size;
      long long prev_local_sum = 0; // local sum of previous band for local difference calculation
      for (int z = 0; z < z_size; z++)
      {
        if (t == 0)
        {
          long long double_resolution_predicted_sample_value = drpsvsmpl->sample(calc_double_resolution_predicted_sample_value(0, 0, z, 0), 0, 0, z);
          long long predicted_sample_value = psvsmpl->sample(calc_predicted_sample_value(double_resolution_predicted_sample_value), 0, 0, z);

          long long prediction_residual = prsmpl->sample(calc_prediction_residual(_image_sample(0, 0, z), predicted_sample_value), 0, 0, z);
          long long quantizer_index = qismpl->sample(calc_quantizer_index(0, 0, prediction_residual), 0, 0, z);

          srsmpl->sample(calc_sample_representative(0, 0, z, 0, double_resolution_predicted_sample_value), 0, 0, z);

          long long theta = tsmpl->sample(calc_theta(0, predicted_sample_value, 0), 0, 0, z);
          mqismpl->sample(calc_mapped_quantizer_index(quantizer_index, theta, double_resolution_predicted_sample_value), 0, 0, z);

          mevsmpl->sample(calc_maximum_error(0, z, predicted_sample_value), 0, 0, z);

          continue;
        }

        // local sum and difference
        long long local_sum = lssmpl->sample(calc_local_sum(x, y, z), y, x, z);
        auto local_difference_vector = calc_local_difference_vector(x, y, z, local_sum, prev_local_sum);
        for (int i = 0; i < local_difference_vector.size(); i++)
          ldvsmpl->sample(local_difference_vector.at(i), y, x, z, i);
        long long predicted_central_local_diff = pcdsmpl->sample(calc_predicted_central_local_diff(x, y, z), y, x, z);

        // prediction calculation
        long long high_resolution_pred_sample_value = hrpsvsmpl->sample(calc_high_resolution_pred_sample_value(x, y, z, local_sum, predicted_central_local_diff), y, x, z);
        long long double_resolution_predicted_sample_value = drpsvsmpl->sample(calc_double_resolution_predicted_sample_value(x, y, z, high_resolution_pred_sample_value), y, x, z);
        long long predicted_sample_value = psvsmpl->sample(calc_predicted_sample_value(double_resolution_predicted_sample_value), y, x, z);

        // quantization
        long long prediction_residual = prsmpl->sample(calc_prediction_residual(_image_sample(y, x, z), predicted_sample_value), y, x, z);
        long long maximum_error = mevsmpl->sample(calc_maximum_error(y, z, predicted_sample_value), y, x, z);
        long long quantizer_index = qismpl->sample(calc_quantizer_index(t, maximum_error, prediction_residual), y, x, z);
        long long clipped_quantizer_bin_center = cqbcsmpl->sample(calc_clipped_quantizer_bin_center(x, y, z, predicted_sample_value, maximum_error, quantizer_index), y, x, z);

        // sample representatives
        long long double_resolution_sample_representative = drsrsmpl->sample(calc_double_resolution_sample_representative(z, clipped_quantizer_bin_center, quantizer_index, maximum_error, high_resolution_pred_sample_value), y, x, z);
        srsmpl->sample(calc_sample_representative(x, y, z, clipped_quantizer_bin_center, double_resolution_sample_representative), y, x, z);
        long long double_resolution_prediction_error = drpesmpl->sample(calc_double_resolution_prediction_error(clipped_quantizer_bin_center, double_resolution_predicted_sample_value), y, x, z);

        // mapping
        long long theta = tsmpl->sample(calc_theta(t, predicted_sample_value, maximum_error), y, x, z);
        mqismpl->sample(calc_mapped_quantizer_index(quantizer_index, theta, double_resolution_predicted_sample_value), y, x, z);

        prev_local_sum = local_sum;

        if (t == t_max)
          continue;

        // weight update
        long long x_weight_update = (t + 1) % x_size;
        long long y_weight_update = (t + 1) / x_size;
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
  local_difference_values_num = header.attr("prediction_bands_num").cast<long long>();

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
    local_difference_values_num += 3;

  weight_component_resolution = header.attr("weight_component_resolution").cast<long long>() + 4;
  weight_update_change_interval = 1LL << (header.attr("weight_update_change_interval").cast<long long>() + 4);
  weight_update_initial_parameter = header.attr("weight_update_initial_parameter").cast<long long>() - 6;
  weight_update_final_parameter = header.attr("weight_update_final_parameter").cast<long long>() - 6;

  std::array<ssize_t, 2> weo_size = {z_size, local_difference_values_num};
  weight_exponent_offset = std::make_unique<Sampler<long long, 2>>(weo_size, 0, true);

  // fill weight exponent offset array
  bool not_all_zero = cast_enum<WeightExponentOffsetFlag>(header.attr("weight_exponent_offset_flag")) == WeightExponentOffsetFlag::NOT_ALL_ZERO;
  if (local_difference_values_num > 0 && not_all_zero)
  {
    PredictionMode mode = cast_enum<PredictionMode>(header.attr("prediction_mode"));
    auto weight_exponent_offset_table = header.attr("weight_exponent_offset_table").cast<NumpyArr<long long>>().unchecked<2>();

    if (mode == PredictionMode::FULL)
    {
      for (int z = 0; z < z_size; z++)
      {
        for (int i = 0; i < 3; i++)
          (*weight_exponent_offset)(z, i) = weight_exponent_offset_table(z, 0);

        for (int i = 3; i < local_difference_values_num; i++)
          (*weight_exponent_offset)(z, i) = weight_exponent_offset_table(z, i - 2);
      }
    }
    else if (mode == PredictionMode::REDUCED)
    {
      for (int z = 0; z < z_size; z++)
        for (int i = 0; i < local_difference_values_num; i++)
          (*weight_exponent_offset)(z, i) = weight_exponent_offset_table(z, i);
    }
  }

  weight_min = -(1LL << (weight_component_resolution + 2));
  weight_max = (1LL << (weight_component_resolution + 2)) - 1;

  register_size = header.attr("register_size").cast<long long>();
  if (register_size == 0)
    register_size = 64;

  std::array<ssize_t, 2> err_size = {y_size, z_size};
  absolute_error_limits = std::make_unique<Sampler<long long, 2>>(err_size, -1, true);
  relative_error_limits = std::make_unique<Sampler<long long, 2>>(err_size, -1, true);

  QuantizerFidelityControlMethod fidelity_control = cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method"));
  bool periodic_error_updating_used = cast_enum<PeriodicErrorUpdatingFlag>(header.attr("periodic_error_updating_flag")) == PeriodicErrorUpdatingFlag::USED;

  // fill error limit arrays
  if (!periodic_error_updating_used && fidelity_control != QuantizerFidelityControlMethod::LOSSLESS)
  {
    if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
    {
      auto absolute_error_limit_table = header.attr("absolute_error_limit_table").cast<NumpyArr<long long>>().unchecked<1>();
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*absolute_error_limits)(y, z) = absolute_error_limit_table(z);
    }
    if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
    {
      auto relative_error_limit_table = header.attr("relative_error_limit_table").cast<NumpyArr<long long>>().unchecked<1>();
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*relative_error_limits)(y, z) = relative_error_limit_table(z);
    }
  }
  else if (periodic_error_updating_used)
  {
    long long period = 1LL << header.attr("error_update_period_exponent").cast<long long>();
    for (long long y = 0; y < (y_size + (1LL << 16) * (long long)(y_size == 0)); y++)
    {
      long long i = y / period;
      if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
      {
        auto periodic_absolute_error_limit_table = header.attr("periodic_absolute_error_limit_table").cast<NumpyArr<long long>>().unchecked<2>();
        for (int z = 0; z < z_size; z++)
          (*absolute_error_limits)(y, z) = periodic_absolute_error_limit_table(i, z);
      }
      if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
      {
        auto periodic_relative_error_limit_table = header.attr("periodic_relative_error_limit_table").cast<NumpyArr<long long>>().unchecked<2>();
        for (int z = 0; z < z_size; z++)
          (*relative_error_limits)(y, z) = periodic_relative_error_limit_table(i, z);
      }
    }
  }
}

void Predictor::init_predictor_arrays()
{
  std::array<ssize_t, 3> image_shape = {image_sample.shape(0), image_sample.shape(1), image_sample.shape(2)};
  std::array<ssize_t, 4> local_difference_vector_shape = {image_shape[0], image_shape[1], image_shape[2], local_difference_values_num};

  assert(image_sample.shape(0) == y_size);
  assert(image_sample.shape(1) == x_size);
  assert(image_sample.shape(2) == z_size);

  // these may be optionally not stored
  lssmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);    // local sum
  pcdsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);   // predicted central local difference
  hrpsvsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates); // high resolution predictied sample value
  drpsvsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates); // long long resolution predicted sample value
  psvsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);   // predicted sample value
  prsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);    // prediction residual
  mevsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);   // maximum error value
  qismpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);    // quantizer index
  cqbcsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);  // clipped quantizer bin center
  drsrsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);  // long long resolution sample representative
  drpesmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);  // long long resolution prediction error
  tsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1, save_intermediates);     // scaled prediction endpoint difference (theta)

  // these must be stored as they are accessed during execution
  mqismpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1);                  // mapped quantizer indices
  srsmpl = std::make_unique<Sampler<long long, 3>>(image_shape, -1);                   // sample representative
  ldvsmpl = std::make_unique<Sampler<long long, 4>>(local_difference_vector_shape, 0); // local difference vectors
  wvsmpl = std::make_unique<Sampler<long long, 4>>(local_difference_vector_shape, 0);  // weight vectors

  init_weights(); // populate the initial weight values

#ifdef DEBUG
  // sorted in order they are calculated
  lssmpl->set_reference("reference/predictor-00-local_sum.csv");                                   //
  ldvsmpl->set_reference("reference/predictor-01-local_difference_vector.csv");                    //
  pcdsmpl->set_reference("reference/predictor-03-predicted_central_local_difference.csv");         //
  hrpsvsmpl->set_reference("reference/predictor-04-high_resolution_predicted_sample_value.csv");   //
  drpsvsmpl->set_reference("reference/predictor-05-double_resolution_predicted_sample_value.csv"); //
  psvsmpl->set_reference("reference/predictor-06-predicted_sample_value.csv");                     //
  prsmpl->set_reference("reference/predictor-07-prediction_residual.csv");                         //
  mevsmpl->set_reference("reference/predictor-22-maximum_error.csv");                              //
  qismpl->set_reference("reference/predictor-09-quantizer_index.csv");                             //
  cqbcsmpl->set_reference("reference/predictor-10-clipped_quantizer_bin_center.csv");              //
  drsrsmpl->set_reference("reference/predictor-11-double_resolution_sample_representative.csv");   //
  srsmpl->set_reference("reference/predictor-12-sample_representative.csv");                       //
  drpesmpl->set_reference("reference/predictor-13-double_resolution_prediction_error.csv");        //
  tsmpl->set_reference("reference/predictor-18-scaled_prediction_endpoint_difference.csv");        //
  mqismpl->set_reference("reference/predictor-14-mapped_quantizer_index.csv");                     //
  wvsmpl->set_reference("reference/predictor-02-weight_vector.csv");                               //
#endif                                                                                             //
}

long long Predictor::calc_local_sum(long long x, long long y, long long z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local sum not defined for t=0");

  long long local_sum = 0;

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
      local_sum = image_constants.attr("middle_sample_value").cast<long long>() * 4;
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
      local_sum = image_constants.attr("middle_sample_value").cast<long long>() * 4;
    break;
  }
  return local_sum;
}

std::vector<long long> Predictor::calc_local_difference_vector(long long x, long long y, long long z, long long local_sum, long long prev_local_sum)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local difference vector not defined for t=0");

  std::vector<long long> local_difference_vector;

  long long offset = 0;

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
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

long long Predictor::calc_predicted_central_local_diff(long long x, long long y, long long z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("predicted central local difference not defined for t=0");

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::REDUCED && z == 0)
    return 0;

  long long acc = 0;
  for (int i = 0; i < local_difference_values_num; i++)
    acc += (*ldvsmpl)(y, x, z, i) * (*wvsmpl)(y, x, z, i);
  return acc;
}

long long Predictor::calc_high_resolution_pred_sample_value(long long x, long long y, long long z, long long local_sum, long long predicted_central_local_diff)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("high resolution predicted sample value not defined for t=0");

  long long sMin = image_constants.attr("lower_sample_limit").cast<long long>();
  long long sMid = image_constants.attr("middle_sample_value").cast<long long>();
  long long sMax = image_constants.attr("upper_sample_limit").cast<long long>();

  long long high_resolution_pred_sample_vaue = modulo_star(
                                                   (predicted_central_local_diff + ((local_sum - (4 * sMid)) << weight_component_resolution)), register_size)
                                               + (sMid
                                                  << (weight_component_resolution + 2))
                                               + (1LL << (weight_component_resolution + 1));

  return std::clamp(high_resolution_pred_sample_vaue, sMin << (weight_component_resolution + 2), (sMax << (weight_component_resolution + 2)) + (1LL << (weight_component_resolution + 1)));
}

long long Predictor::calc_double_resolution_predicted_sample_value(long long x, long long y, long long z, long long high_resolution_pred_sample_value)
{
  long long double_resolution_predicted_sample_value;
  long long P = header.attr("prediction_bands_num").cast<long long>();

  if ((x == 0 && y == 0) && P > 0 && z > 0)
    double_resolution_predicted_sample_value = 2 * image_sample.at(y, x, z - 1);
  else if (x == 0 && y == 0 && (P == 0 || z == 0))
    double_resolution_predicted_sample_value = 2 * image_constants.attr("middle_sample_value").cast<long long>();
  else if (x != 0 || y != 0)
    double_resolution_predicted_sample_value = high_resolution_pred_sample_value >> (weight_component_resolution + 1);

  return double_resolution_predicted_sample_value;
}

long long Predictor::calc_predicted_sample_value(long long double_resolution_predicted_sample_value)
{
  return double_resolution_predicted_sample_value >> 1;
}

long long Predictor::calc_prediction_residual(long long sample, long long predicted_sample_value)
{
  return sample - predicted_sample_value;
}

long long Predictor::calc_maximum_error(long long y, long long z, long long predicted_sample_value)
{
  long long maximum_error;

  switch (cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method")))
  {
  case QuantizerFidelityControlMethod::LOSSLESS:
    maximum_error = 0;
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_ONLY:
    maximum_error = (*absolute_error_limits)(y, z);
    break;

  case QuantizerFidelityControlMethod::RELATIVE_ONLY:
    maximum_error = (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<long long>();
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_AND_RELATIVE:
    maximum_error = std::min((*absolute_error_limits)(y, z), (*relative_error_limits)(y, z) * predicted_sample_value / image_constants.attr("dynamic_range").cast<long long>());
    break;
  }

  return maximum_error;
}

long long Predictor::calc_quantizer_index(long long t, long long maximum_error, long long prediction_residual)
{
  if (t == 0)
    return prediction_residual;
  else
    return sgn(prediction_residual) * (std::abs(prediction_residual) + maximum_error) / (2 * maximum_error + 1);
}

long long Predictor::calc_clipped_quantizer_bin_center(long long x, long long y, long long z, long long predicted_sample_value, long long maximum_error, long long quantizer_index)
{
  long long sMin = image_constants.attr("lower_sample_limit").cast<long long>();
  long long sMax = image_constants.attr("upper_sample_limit").cast<long long>();

  if (maximum_error == 0)
    return image_sample.at(y, x, z);

  return std::clamp(predicted_sample_value + quantizer_index * (2 * maximum_error + 1), sMin, sMax);
}

long long Predictor::calc_double_resolution_sample_representative(long long z, long long clipped_quantizer_bin_center, long long quantizer_index, long long maximum_error, long long high_resolution_pred_sample_value)
{
  auto phi = header.attr("damping_table_array")
                 .cast<NumpyArr<long long>>()
                 .at(z);

  auto psi = header.attr("damping_offset_table_array")
                 .cast<NumpyArr<long long>>()
                 .at(z);

  long long Theta = header.attr("sample_representative_resolution").cast<long long>();
  long long Omega = weight_component_resolution;

  // Shortcut from spec
  if (phi == 0 && psi == 0)
    return 2 * clipped_quantizer_bin_center;

  // Precompute powers of two
  long long two_Theta = 1LL << Theta;
  long long two_Omega = 1LL << Omega;
  long long two_Omega_minus_Theta = 1LL << (Omega - Theta);
  long long two_Omega_plus_1 = 1LL << (Omega + 1);
  long long denom = 1LL << (Omega + Theta + 1);

  // s′z(t) · 2^Ω
  long long term_signal = clipped_quantizer_bin_center * two_Omega;

  // sgn(qz(t)) · mz(t) · ψz · 2^(Ω−Θ)
  long long term_error =
      sgn(quantizer_index) * maximum_error * psi * two_Omega_minus_Theta;

  // 4 · (2^Θ − φz)
  long long gain = 4 * (two_Theta - phi);

  // Full numerator of Eq. (47)
  long long numerator =
      gain * (term_signal - term_error)
      + phi * high_resolution_pred_sample_value
      - phi * two_Omega_plus_1;

  // Divide by 2^(Ω+Θ+1)
  return numerator / denom;
}

long long Predictor::calc_sample_representative(long long x, long long y, long long z, long long clipped_quantizer_bin_center, long long double_resolution_sample_representative)
{
  if (x == 0 && y == 0)
    return image_sample.at(y, x, z);

  auto damping_table_array = header.attr("damping_table_array").cast<NumpyArr<long long>>().unchecked<1>();
  auto damping_offset_table_array = header.attr("damping_offset_table_array").cast<NumpyArr<long long>>().unchecked<1>();

  if (damping_table_array(0) == 0 && damping_offset_table_array(0) == 0)
    return clipped_quantizer_bin_center;

  return (double_resolution_sample_representative + 1) / 2;
}

long long Predictor::calc_double_resolution_prediction_error(long long clipped_quantizer_bin_center, long long double_resolution_predicted_sample_value)
{
  return 2 * clipped_quantizer_bin_center - double_resolution_predicted_sample_value;
}

long long Predictor::calc_theta(long long t, long long predicted_sample_value, long long maximum_error)
{
  long long sMin = image_constants.attr("lower_sample_limit").cast<long long>();
  long long sMax = image_constants.attr("upper_sample_limit").cast<long long>();

  if (t == 0)
    return std::min(predicted_sample_value - sMin, sMax - predicted_sample_value);

  long long denominator = 2 * maximum_error + 1;

  return std::min((predicted_sample_value - sMin + maximum_error) / denominator, (sMax - predicted_sample_value + maximum_error) / denominator);
}

long long Predictor::calc_mapped_quantizer_index(long long quantizer_index, long long theta, long long double_resolution_predicted_sample_value)
{
  long long sign = (double_resolution_predicted_sample_value & 1) ? -1 : 1;
  long long term = sign * quantizer_index;

  if (std::abs(quantizer_index) > theta)
    return std::abs(quantizer_index) + theta;
  else if (0 <= term && term <= theta)
    return 2 * std::abs(quantizer_index);
  else
    return 2 * std::abs(quantizer_index) - 1;
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

    for (long long z = 1; z < z_size; z++)
    {
      if (SPECTRAL_BANDS_USED(z) == 0)
        continue;

      (*wvsmpl)(y, x, z, offset) = (1LL << weight_component_resolution) * 7 / 8;

      for (int i = 1; i < SPECTRAL_BANDS_USED(z); i++)
        (*wvsmpl)(y, x, z, offset + i) = (*wvsmpl)(y, x, z, offset + i - 1) / 8;
    }
  }
  else
  {
    long long multiplier = 1LL << (weight_component_resolution + 3 - header.attr("weight_init_resolution").cast<long long>());
    long long offset = std::ceil((1LL << (weight_component_resolution + 2 - header.attr("weight_init_resolution").cast<long long>())) - 1);

    auto weight_init_table = header.attr("weight_init_table").cast<NumpyArr<long long>>().unchecked<2>();

    for (long long z = 0; z < z_size; z++)
      for (int i = 0; i < local_difference_values_num; i++)
        (*wvsmpl)(y, x, z, i) = multiplier * weight_init_table(z, i) + offset;
  }
}

long long calc_weight_offset(long long local_diff, long long weight_update_scaling_exponent, long long weight_exponent_offset, long long double_resolution_prediction_error)
{
  long long exponent = weight_update_scaling_exponent + weight_exponent_offset;
  if (exponent > 0)
    return ((((sgn_positive(double_resolution_prediction_error) * local_diff) >> exponent) + 1) >> 1);
  else
    return ((((sgn_positive(double_resolution_prediction_error) * local_diff) << (-exponent)) + 1) >> 1);
}

std::vector<long long> Predictor::calc_weight_vector(long long x, long long y, long long z, long long double_resolution_prediction_error)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("weight vector not defined for t=0");

  long long t = x + y * x_size;
  long long weight_update_scaling_exponent = std::clamp(weight_update_initial_parameter + (t - x_size) / weight_update_change_interval,
                                                        weight_update_initial_parameter,
                                                        weight_update_final_parameter)
                                             + image_constants.attr("dynamic_range_bits").cast<long long>()
                                             - weight_component_resolution;

  std::vector<long long> weight_vector(local_difference_values_num);

  // calculates weight vector for t+1
  for (int i = 0; i < weight_vector.size(); i++)
  {
    long long local_diff = (*ldvsmpl)(y, x, z, i);
    long long weo = (*weight_exponent_offset)(z, i);

    long long weight_offset = calc_weight_offset(local_diff, weight_update_scaling_exponent, weo, double_resolution_prediction_error);
    long long weight_unclipped = (*wvsmpl)(y, x, z, i) + weight_offset;

    weight_vector.at(i) = std::clamp(weight_unclipped, weight_min, weight_max);
  }

  return weight_vector;
}
