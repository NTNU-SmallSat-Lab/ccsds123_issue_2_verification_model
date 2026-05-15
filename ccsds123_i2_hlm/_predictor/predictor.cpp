#include "predictor.hpp"
#include "header_types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#define SPECTRAL_BANDS_USED(z) std::min(z, header.attr("prediction_bands_num").cast<ll>()) // symbol P^*

/******************** Utils ********************/

// cast Python enum to C++ enum class
template <typename T>
static inline T cast_enum(py::object enum_py)
{
  auto value = enum_py.attr("value").cast<ll>();
  return static_cast<T>(value);
}

static inline long long modulo_star(long long value, long long r)
{
  if (r == 64)
    return value;

  long long offset = 1LL << (r - 1);
  long long modulus = 1LL << r;

  long long wrapped = (value + offset) % modulus;
  if (wrapped < 0) // cpp and python treats modulo of negative numbers differently
    wrapped += modulus;

  return wrapped - offset;
}

static inline ll sgn(ll value)
{
  if (value < 0)
    return -1;
  if (value > 0)
    return 1;
  return 0;
}

static inline ll sgn_positive(ll value)
{
  if (value >= 0)
    return 1;
  return -1;
}

static inline ll floor_div2(ll x)
{
  ll q = x / 2;
  ll r = x % 2;

  if (r != 0 && x < 0)
    --q;

  return q;
}

/******************** (de)Constructor ********************/

Predictor::Predictor(py::object header, py::object image_constants, bool delayed_weight_updates, bool save_intermediates)
    : header(header),
      image_constants(image_constants),
      delayed_weight_updates(delayed_weight_updates),
      save_intermediates(save_intermediates)
{
  x_size = header.attr("x_size").cast<ll>();
  y_size = header.attr("y_size").cast<ll>();
  z_size = header.attr("z_size").cast<ll>();

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

NumpyArr<ll> Predictor::compress(NumpyArr<ll> image_sample)
{
  ll t_max = x_size * y_size - 1;
  auto _image_sample = image_sample.unchecked<3>();

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++)
  {
    std::cout << "\rProcessing frame y=" << y + 1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++)
    {
      int t = x + y * x_size;
      ll prev_ls = 0; // local sum of previous band for local difference calculation
      for (int z = 0; z < z_size; z++)
      {
        if (t == 0)
        {
          ll drpsv = drpsvsmpl->sample(calc_drpsv(0, 0, z, 0, image_sample), 0, 0, z);
          ll psv = psvsmpl->sample(calc_psv(drpsv), 0, 0, z);

          ll pr = prsmpl->sample(calc_pr(_image_sample(0, 0, z), psv), 0, 0, z);
          ll qi = qismpl->sample(calc_qi(0, 0, pr), 0, 0, z);

          srsmpl->sample(calc_sample_representative(0, 0, z, 0, drpsv, image_sample), 0, 0, z);

          ll theta = tsmpl->sample(calc_theta(0, psv, 0), 0, 0, z);
          mqismpl->sample(calc_mqi(qi, theta, drpsv), 0, 0, z);

          mevsmpl->sample(calc_mev(0, z, psv), 0, 0, z);

          continue;
        }

        // local sum and difference
        ll ls = lssmpl->sample(calc_ls(x, y, z), y, x, z);
        auto ldv = calc_ldv(x, y, z, ls, prev_ls);
        for (int i = 0; i < ldv.size(); i++)
          ldvsmpl->sample(ldv.at(i), y, x, z, i);

        // prediction calculation
        ll pcd = pcdsmpl->sample(calc_pcd(x, y, z), y, x, z);
        ll hrpsv = hrpsvsmpl->sample(calc_hrpsv(x, y, z, ls, pcd), y, x, z);
        ll drpsv = drpsvsmpl->sample(calc_drpsv(x, y, z, hrpsv, image_sample), y, x, z);
        ll psv = psvsmpl->sample(calc_psv(drpsv), y, x, z);

        // quantization
        ll pr = prsmpl->sample(calc_pr(_image_sample(y, x, z), psv), y, x, z);
        ll mev = mevsmpl->sample(calc_mev(y, z, psv), y, x, z);
        ll qi = qismpl->sample(calc_qi(t, mev, pr), y, x, z);

        // sample representatives
        ll cqbc = cqbcsmpl->sample(calc_cqbc(x, y, z, psv, mev, qi, image_sample), y, x, z);
        ll drsr = drsrsmpl->sample(calc_drsr(z, cqbc, qi, mev, hrpsv), y, x, z);
        srsmpl->sample(calc_sample_representative(x, y, z, cqbc, drsr, image_sample), y, x, z);
        drpesmpl->sample(calc_drpe(cqbc, drpsv), y, x, z); // needed by weight update

        // mapping
        ll theta = tsmpl->sample(calc_theta(t, psv, mev), y, x, z);
        mqismpl->sample(calc_mqi(qi, theta, drpsv), y, x, z);

        prev_ls = ls;

        if (t == t_max)
          continue;

        // weight update
        ll x_weight_update = (t + 1) % x_size;
        ll y_weight_update = (t + 1) / x_size;
        assert(x_weight_update < x_size && y_weight_update < y_size);

        auto updated_weight_vector = calc_weight_vector(x, y, z);
        for (int i = 0; i < updated_weight_vector.size(); i++) // sets weights for t+1
          wvsmpl->sample(updated_weight_vector.at(i), y_weight_update, x_weight_update, z, i);
      }
    }
  }
  std::cout << std::endl;

  return mqismpl->get_arr();
}

NumpyArr<ll> Predictor::decompress(NumpyArr<ll> mqi)
{
  auto decompressed_image_sample = NumpyArr<ll>({y_size, x_size, z_size});
  auto _decompressed_image_sample = decompressed_image_sample.mutable_unchecked<3>();
  auto _mqi = mqi.mutable_unchecked<3>();

  ll t_max = x_size * y_size - 1;

  // tranversing in BIP order
  for (int y = 0; y < y_size; y++)
  {
    std::cout << "\rProcessing frame y=" << y + 1 << "/" << y_size << std::flush;
    for (int x = 0; x < x_size; x++)
    {
      int t = x + y * x_size;
      ll prev_ls = 0; // local sum of previous band for local difference calculation
      for (int z = 0; z < z_size; z++)
      {
        if (t == 0)
        {
          ll drpsv = drpsvsmpl->sample(calc_drpsv(0, 0, z, 0, decompressed_image_sample), 0, 0, z);
          ll psv = psvsmpl->sample(calc_psv(drpsv), 0, 0, z);

          ll theta = tsmpl->sample(calc_theta(0, psv, 0), 0, 0, z);

          ll mev = mevsmpl->sample(calc_mev(0, z, psv), 0, 0, z);
          ll qi = qismpl->sample(decalc_qi(theta, _mqi(0, 0, z), psv, drpsv), 0, 0, z);

          ll pr = prsmpl->sample(decalc_pr(t, qi, mev), 0, 0, z);

          _decompressed_image_sample(0, 0, z) = decalc_sample(pr, psv);
          srsmpl->sample(calc_sample_representative(x, y, z, 0, 0, decompressed_image_sample), 0, 0, z);

          continue;
        }

        // local sum and difference
        ll ls = lssmpl->sample(calc_ls(x, y, z), y, x, z);
        auto ldv = calc_ldv(x, y, z, ls, prev_ls);
        for (int i = 0; i < ldv.size(); i++)
          ldvsmpl->sample(ldv.at(i), y, x, z, i);

        // prediction calculation
        ll pcd = pcdsmpl->sample(calc_pcd(x, y, z), y, x, z);
        ll hrpsv = hrpsvsmpl->sample(calc_hrpsv(x, y, z, ls, pcd), y, x, z);
        ll drpsv = drpsvsmpl->sample(calc_drpsv(x, y, z, hrpsv, decompressed_image_sample), y, x, z);
        ll psv = psvsmpl->sample(calc_psv(drpsv), y, x, z);

        // decompression
        ll mev = mevsmpl->sample(calc_mev(y, z, psv), y, x, z);
        ll theta = tsmpl->sample(calc_theta(t, psv, mev), y, x, z);
        ll qi = qismpl->sample(decalc_qi(theta, _mqi(y, x, z), psv, drpsv), y, x, z);
        ll pr = prsmpl->sample(decalc_pr(t, qi, mev), y, x, z);
        _decompressed_image_sample(y, x, z) = decalc_sample(pr, psv);

        // sample representatives
        ll cqbc = cqbcsmpl->sample(calc_cqbc(x, y, z, psv, mev, qi, decompressed_image_sample), y, x, z);
        ll drsr = drsrsmpl->sample(calc_drsr(z, cqbc, qi, mev, hrpsv), y, x, z);
        srsmpl->sample(calc_sample_representative(x, y, z, cqbc, drsr, decompressed_image_sample), y, x, z);
        drpesmpl->sample(calc_drpe(cqbc, drpsv), y, x, z); // needed by weight update

        prev_ls = ls;

        if (t == t_max)
          continue;

        // weight update
        ll x_weight_update = (t + 1) % x_size;
        ll y_weight_update = (t + 1) / x_size;
        assert(x_weight_update < x_size && y_weight_update < y_size);

        auto updated_weight_vector = calc_weight_vector(x, y, z);
        for (int i = 0; i < updated_weight_vector.size(); i++) // sets weights for t+1
          wvsmpl->sample(updated_weight_vector.at(i), y_weight_update, x_weight_update, z, i);
      }
    }
  }
  std::cout << std::endl;

  return decompressed_image_sample;
}

void Predictor::save_data(std::string output_folder)
{
  if (!save_intermediates)
    return;

  // import numpy for easy storing to file
  py::object numpy = py::module_::import("numpy");
  py::object savetxt = numpy.attr("savetxt");

  auto csv_image_shape = {y_size * x_size, z_size};
  auto csv_vector_shape = {y_size * x_size, z_size * local_difference_values_num};

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
  local_difference_values_num = header.attr("prediction_bands_num").cast<ll>();

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
    local_difference_values_num += 3;

  weight_component_resolution = header.attr("weight_component_resolution").cast<ll>() + 4;
  weight_update_change_interval = 1LL << (header.attr("weight_update_change_interval").cast<ll>() + 4);
  weight_update_initial_parameter = header.attr("weight_update_initial_parameter").cast<ll>() - 6;
  weight_update_final_parameter = header.attr("weight_update_final_parameter").cast<ll>() - 6;

  std::array<ssize_t, 2> weo_size = {z_size, local_difference_values_num};
  weight_exponent_offset = std::make_unique<Sampler<ll, 2>>(weo_size, 0, true);

  // fill weight exponent offset array
  bool not_all_zero = cast_enum<WeightExponentOffsetFlag>(header.attr("weight_exponent_offset_flag")) == WeightExponentOffsetFlag::NOT_ALL_ZERO;
  if (local_difference_values_num > 0 && not_all_zero)
  {
    PredictionMode mode = cast_enum<PredictionMode>(header.attr("prediction_mode"));
    auto weight_exponent_offset_table = header.attr("weight_exponent_offset_table").cast<NumpyArr<ll>>().unchecked<2>();

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

  register_size = header.attr("register_size").cast<ll>();
  if (register_size == 0)
    register_size = 64;

  std::array<ssize_t, 2> err_size = {y_size, z_size};
  absolute_error_limits = std::make_unique<Sampler<ll, 2>>(err_size, -1, true);
  relative_error_limits = std::make_unique<Sampler<ll, 2>>(err_size, -1, true);

  QuantizerFidelityControlMethod fidelity_control = cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method"));
  bool periodic_error_updating_used = cast_enum<PeriodicErrorUpdatingFlag>(header.attr("periodic_error_updating_flag")) == PeriodicErrorUpdatingFlag::USED;

  // fill error limit arrays
  if (!periodic_error_updating_used && fidelity_control != QuantizerFidelityControlMethod::LOSSLESS)
  {
    if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
    {
      auto absolute_error_limit_table = header.attr("absolute_error_limit_table").cast<NumpyArr<ll>>().unchecked<1>();
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*absolute_error_limits)(y, z) = absolute_error_limit_table(z);
    }
    if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
    {
      auto relative_error_limit_table = header.attr("relative_error_limit_table").cast<NumpyArr<ll>>().unchecked<1>();
      for (int y = 0; y < y_size; y++)
        for (int z = 0; z < z_size; z++)
          (*relative_error_limits)(y, z) = relative_error_limit_table(z);
    }
  }
  else if (periodic_error_updating_used)
  {
    ll period = 1LL << header.attr("error_update_period_exponent").cast<ll>();
    for (ll y = 0; y < (y_size + (1LL << 16) * (ll)(y_size == 0)); y++)
    {
      ll i = y / period;
      if (fidelity_control != QuantizerFidelityControlMethod::RELATIVE_ONLY)
      {
        auto periodic_absolute_error_limit_table = header.attr("periodic_absolute_error_limit_table").cast<NumpyArr<ll>>().unchecked<2>();
        for (int z = 0; z < z_size; z++)
          (*absolute_error_limits)(y, z) = periodic_absolute_error_limit_table(i, z);
      }
      if (fidelity_control != QuantizerFidelityControlMethod::ABSOLUTE_ONLY)
      {
        auto periodic_relative_error_limit_table = header.attr("periodic_relative_error_limit_table").cast<NumpyArr<ll>>().unchecked<2>();
        for (int z = 0; z < z_size; z++)
          (*relative_error_limits)(y, z) = periodic_relative_error_limit_table(i, z);
      }
    }
  }
}

void Predictor::init_predictor_arrays()
{
  std::array<ssize_t, 3> image_shape = {y_size, x_size, z_size};
  std::array<ssize_t, 4> ldv_shape = {image_shape[0], image_shape[1], image_shape[2], local_difference_values_num};

  // these may be optionally not stored
  lssmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);    // local sum
  pcdsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);   // predicted central local difference
  hrpsvsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates); // high resolution predictied sample value
  drpsvsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates); // double resolution predicted sample value
  psvsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);   // predicted sample value
  prsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);    // prediction residual
  mevsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);   // maximum error value
  qismpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);    // quantizer index
  cqbcsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);  // clipped quantizer bin center
  drsrsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);  // double resolution sample representative
  tsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1, save_intermediates);     // scaled prediction endpoint difference (theta)

  // these must be stored as they are accessed during execution
  drpesmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1); // double resolution prediction error
  mqismpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1);  // mapped quantizer indices
  srsmpl = std::make_unique<Sampler<ll, 3>>(image_shape, -1);   // sample representative
  ldvsmpl = std::make_unique<Sampler<ll, 4>>(ldv_shape, 0);     // local difference vectors
  wvsmpl = std::make_unique<Sampler<ll, 4>>(ldv_shape, 0);      // weight vectors

  init_weights(); // populate the initial weight values

#ifdef DEBUG
  // sorted in order they are calculated
  // all should be the same for decompression/compression
  // except for prediction residual in near-lossless mode
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

ll Predictor::calc_ls(ll x, ll y, ll z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local sum not defined for t=0");

  ll ls = 0;

  switch (cast_enum<LocalSumType>(header.attr("local_sum_type")))
  {

  case LocalSumType::WIDE_NEIGHBOR_ORIENTED:
    if (y > 0 && 0 < x && x < x_size - 1)
      ls = (*srsmpl)(y, x - 1, z) + (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z);
    else if (y == 0 && x > 0)
      ls = (*srsmpl)(y, x - 1, z) * 4;
    else if (y > 0 && x == 0)
      ls = ((*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z)) * 2;
    else if (y > 0 && x == x_size - 1)
      ls = (*srsmpl)(y, x - 1, z) + (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) * 2;
    break;

  case LocalSumType::NARROW_NEIGHBOR_ORIENTED:
    if (y > 0 && 0 < x && x < x_size - 1)
      ls = (*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z) * 2 + (*srsmpl)(y - 1, x + 1, z);
    else if (y == 0 && x > 0 && z > 0)
      ls = (*srsmpl)(y, x - 1, z - 1) * 4;
    else if (y > 0 && x == 0)
      ls = ((*srsmpl)(y - 1, x, z) + (*srsmpl)(y - 1, x + 1, z)) * 2;
    else if (y > 0 && x == x_size - 1)
      ls = ((*srsmpl)(y - 1, x - 1, z) + (*srsmpl)(y - 1, x, z)) * 2;
    else if (y == 0 && x > 0 && z == 0)
      ls = image_constants.attr("middle_sample_value").cast<ll>() * 4;
    break;

  case LocalSumType::WIDE_COLUMN_ORIENTED:
    if (y > 0)
      ls = (*srsmpl)(y - 1, x, z) * 4;
    else if (y == 0 && x > 0)
      ls = (*srsmpl)(y, x - 1, z) * 4;
    break;

  case LocalSumType::NARROW_COLUMN_ORIENTED:
    if (y > 0)
      ls = (*srsmpl)(y - 1, x, z) * 4;
    else if (y == 0 && x > 0 && z > 0)
      ls = (*srsmpl)(y, x - 1, z - 1) * 4;
    else if (y == 0 && x > 0 && z == 0)
      ls = image_constants.attr("middle_sample_value").cast<ll>() * 4;
    break;
  }
  return ls;
}

std::vector<ll> Predictor::calc_ldv(ll x, ll y, ll z, ll ls, ll prev_ls)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("local difference vector not defined for t=0");

  std::vector<ll> ldv;

  ll offset = 0;

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
  {
    for (int i = 0; i < 3; i++)
      ldv.push_back(0);

    if (x > 0 && y > 0)
    {
      ldv.at(0) = 4 * (*srsmpl)(y - 1, x, z) - ls;
      ldv.at(1) = 4 * (*srsmpl)(y, x - 1, z) - ls;
      ldv.at(2) = 4 * (*srsmpl)(y - 1, x - 1, z) - ls;
    }
    else if (x == 0 && y > 0)
    {
      ldv.at(0) = 4 * (*srsmpl)(y - 1, x, z) - ls;
      ldv.at(1) = 4 * (*srsmpl)(y - 1, x, z) - ls;
      ldv.at(2) = 4 * (*srsmpl)(y - 1, x, z) - ls;
    }

    offset += 3;
  }

  if (z > 0 && SPECTRAL_BANDS_USED(z) > 0)
  {
    // local difference of z-1
    ldv.push_back(4 * (*srsmpl)(y, x, z - 1) - prev_ls);

    // copy local differences of previous vector
    for (int i = 1; i < SPECTRAL_BANDS_USED(z); i++)
      ldv.push_back((*ldvsmpl)(y, x, z - 1, offset + i - 1));
  }

  return ldv;
}

ll Predictor::calc_pcd(ll x, ll y, ll z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("predicted central local difference not defined for t=0");

  if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::REDUCED && z == 0)
    return 0;

  ll acc = 0;
  for (int i = 0; i < local_difference_values_num; i++)
    acc += (*ldvsmpl)(y, x, z, i) * (*wvsmpl)(y, x, z, i);
  return acc;
}

ll Predictor::calc_hrpsv(ll x, ll y, ll z, ll ls, ll pcd)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("high resolution predicted sample value not defined for t=0");

  ll sMin = image_constants.attr("lower_sample_limit").cast<ll>();
  ll sMid = image_constants.attr("middle_sample_value").cast<ll>();
  ll sMax = image_constants.attr("upper_sample_limit").cast<ll>();

  ll high_resolution_pred_sample_vaue = modulo_star(
                                            (pcd + ((ls - (4 * sMid)) << weight_component_resolution)), register_size)
                                        + (sMid
                                           << (weight_component_resolution + 2))
                                        + (1LL << (weight_component_resolution + 1));

  return std::clamp(high_resolution_pred_sample_vaue, sMin << (weight_component_resolution + 2), (sMax << (weight_component_resolution + 2)) + (1LL << (weight_component_resolution + 1)));
}

ll Predictor::calc_drpsv(ll x, ll y, ll z, ll hrpsv, NumpyArr<ll> image_sample)
{
  ll drpsv;
  ll P = header.attr("prediction_bands_num").cast<ll>();

  if ((x == 0 && y == 0) && P > 0 && z > 0)
    drpsv = 2 * image_sample.at(y, x, z - 1);
  else if (x == 0 && y == 0 && (P == 0 || z == 0))
    drpsv = 2 * image_constants.attr("middle_sample_value").cast<ll>();
  else if (x != 0 || y != 0)
    drpsv = hrpsv >> (weight_component_resolution + 1);

  return drpsv;
}

ll Predictor::calc_psv(ll drpsv)
{
  return drpsv >> 1;
}

ll Predictor::calc_pr(ll sample, ll psv)
{
  return sample - psv;
}

ll Predictor::calc_mev(ll y, ll z, ll psv)
{
  ll mev;

  switch (cast_enum<QuantizerFidelityControlMethod>(header.attr("quantizer_fidelity_control_method")))
  {
  case QuantizerFidelityControlMethod::LOSSLESS:
    mev = 0;
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_ONLY:
    mev = (*absolute_error_limits)(y, z);
    break;

  case QuantizerFidelityControlMethod::RELATIVE_ONLY:
    mev = (*relative_error_limits)(y, z) * psv / image_constants.attr("dynamic_range").cast<ll>();
    break;

  case QuantizerFidelityControlMethod::ABSOLUTE_AND_RELATIVE:
    mev = std::min((*absolute_error_limits)(y, z), (*relative_error_limits)(y, z) * psv / image_constants.attr("dynamic_range").cast<ll>());
    break;
  }

  return mev;
}

ll Predictor::calc_qi(ll t, ll mev, ll pr)
{
  if (t == 0)
    return pr;
  else
    return sgn(pr) * (std::abs(pr) + mev) / (2 * mev + 1);
}

ll Predictor::calc_cqbc(ll x, ll y, ll z, ll psv, ll mev, ll qi, NumpyArr<ll> image_sample)
{
  ll sMin = image_constants.attr("lower_sample_limit").cast<ll>();
  ll sMax = image_constants.attr("upper_sample_limit").cast<ll>();

  if (mev == 0)
    return image_sample.at(y, x, z);

  return std::clamp(psv + qi * (2 * mev + 1), sMin, sMax);
}

ll Predictor::calc_drsr(ll z, ll cqbc, ll qi, ll mev, ll hrpsv)
{
  auto phi = header.attr("damping_table_array")
                 .cast<NumpyArr<ll>>()
                 .at(z);

  auto psi = header.attr("damping_offset_table_array")
                 .cast<NumpyArr<ll>>()
                 .at(z);

  ll Theta = header.attr("sample_representative_resolution").cast<ll>();
  ll Omega = weight_component_resolution;

  // Shortcut from spec
  if (phi == 0 && psi == 0)
    return 2 * cqbc;

  // Precompute powers of two
  ll two_Theta = 1LL << Theta;
  ll two_Omega = 1LL << Omega;
  ll two_Omega_minus_Theta = 1LL << (Omega - Theta);
  ll two_Omega_plus_1 = 1LL << (Omega + 1);
  ll denom = 1LL << (Omega + Theta + 1);

  // s′z(t) · 2^Ω
  ll term_signal = cqbc * two_Omega;

  // sgn(qz(t)) · mz(t) · ψz · 2^(Ω−Θ)
  ll term_error =
      sgn(qi) * mev * psi * two_Omega_minus_Theta;

  // 4 · (2^Θ − φz)
  ll gain = 4 * (two_Theta - phi);

  // Full numerator of Eq. (47)
  ll numerator =
      gain * (term_signal - term_error)
      + phi * hrpsv
      - phi * two_Omega_plus_1;

  // Divide by 2^(Ω+Θ+1)
  return numerator / denom;
}

ll Predictor::calc_sample_representative(ll x, ll y, ll z, ll cqbc, ll drsr, NumpyArr<ll> image_sample)
{
  if (x == 0 && y == 0)
    return image_sample.at(y, x, z);

  auto damping_table_array = header.attr("damping_table_array").cast<NumpyArr<ll>>().unchecked<1>();
  auto damping_offset_table_array = header.attr("damping_offset_table_array").cast<NumpyArr<ll>>().unchecked<1>();

  if (damping_table_array(z) == 0 && damping_offset_table_array(z) == 0)
    return cqbc;

  return (drsr + 1) / 2;
}

ll Predictor::calc_drpe(ll cqbc, ll drpsv)
{
  return 2 * cqbc - drpsv;
}

ll Predictor::calc_theta(ll t, ll psv, ll mev)
{
  ll sMin = image_constants.attr("lower_sample_limit").cast<ll>();
  ll sMax = image_constants.attr("upper_sample_limit").cast<ll>();

  if (t == 0)
    return std::min(psv - sMin, sMax - psv);

  ll denominator = 2 * mev + 1;

  return std::min((psv - sMin + mev) / denominator, (sMax - psv + mev) / denominator);
}

ll Predictor::calc_mqi(ll qi, ll theta, ll drpsv)
{
  ll sign = (drpsv & 1) ? -1 : 1;
  ll term = sign * qi;

  if (std::abs(qi) > theta)
    return std::abs(qi) + theta;
  else if (0 <= term && term <= theta)
    return 2 * std::abs(qi);
  else
    return 2 * std::abs(qi) - 1;
}

void Predictor::init_weights()
{
  if (x_size == 1 && y_size == 1)
    return; // weights should be all 0

  // set x and y corresponding to t=1
  int x = 0;
  int y = 0;
  if (x_size == 1)
    y = 1;
  else
    x = 1;

  if (cast_enum<WeightInitMethod>(header.attr("weight_init_method")) == WeightInitMethod::DEFAULT)
  {
    int offset = 0;

    if (cast_enum<PredictionMode>(header.attr("prediction_mode")) == PredictionMode::FULL)
      offset += 3; // N, W and NW are already zero initialized

    for (ll z = 1; z < z_size; z++)
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
    ll multiplier = 1LL << (weight_component_resolution + 3 - header.attr("weight_init_resolution").cast<ll>());
    ll offset = std::ceil((1LL << (weight_component_resolution + 2 - header.attr("weight_init_resolution").cast<ll>())) - 1);

    auto weight_init_table = header.attr("weight_init_table").cast<NumpyArr<ll>>().unchecked<2>();

    for (ll z = 0; z < z_size; z++)
      for (int i = 0; i < local_difference_values_num; i++)
        (*wvsmpl)(y, x, z, i) = multiplier * weight_init_table(z, i) + offset;
  }
}

ll calc_weight_offset(ll local_diff, ll weight_update_scaling_exponent, ll weight_exponent_offset, ll drpe)
{
  ll exponent = weight_update_scaling_exponent + weight_exponent_offset;
  if (exponent > 0)
    return ((((sgn_positive(drpe) * local_diff) >> exponent) + 1) >> 1);
  else
    return ((((sgn_positive(drpe) * local_diff) << (-exponent)) + 1) >> 1);
}

std::vector<ll> Predictor::calc_weight_vector(ll x, ll y, ll z)
{
  if (x == 0 && y == 0)
    throw std::invalid_argument("weight vector not defined for t=0");

  ll t = x + y * x_size;
  ll t_weight_update = delayed_weight_updates ? t - 3 : t;
  ll x_weight_update = delayed_weight_updates ? t_weight_update % x_size : x;
  ll y_weight_update = delayed_weight_updates ? t_weight_update / x_size : y;

  ll weight_update_scaling_exponent = std::clamp(weight_update_initial_parameter + (t_weight_update - x_size) / weight_update_change_interval,
                                                 weight_update_initial_parameter,
                                                 weight_update_final_parameter)
                                      + image_constants.attr("dynamic_range_bits").cast<ll>()
                                      - weight_component_resolution;

  std::vector<ll> weight_vector(local_difference_values_num);

  // calculates weight vector for t+1
  for (int i = 0; i < weight_vector.size(); i++)
  {
    if (delayed_weight_updates && x < 3)
    { // keep weight
      weight_vector.at(i) = (*wvsmpl)(y, x, z, i);
      continue;
    }

    ll local_diff = (*ldvsmpl)(y_weight_update, x_weight_update, z, i);
    ll drpe = (*drpesmpl)(y_weight_update, x_weight_update, z);
    ll weo = (*weight_exponent_offset)(z, i);

    ll weight_offset = calc_weight_offset(local_diff, weight_update_scaling_exponent, weo, drpe);
    ll weight_unclipped = (*wvsmpl)(y_weight_update, x_weight_update, z, i) + weight_offset;

    if (delayed_weight_updates && x > 3)
    { // refine weight
      weight_unclipped = weight_unclipped + (*wvsmpl)(y, x, z, i);
      weight_unclipped = floor_div2(weight_unclipped); // to avoid weird rounding errors of negative numbers
    }

    weight_vector.at(i) = std::clamp(weight_unclipped, weight_min, weight_max);
  }

  return weight_vector;
}

ll Predictor::decalc_qi(ll theta, ll mqi, ll psv, ll drpsv)
{
  if (mqi > 2 * theta)
  {
    ll sMid = image_constants.attr("middle_sample_value").cast<ll>();
    return (theta - mqi) * sgn_positive(psv - sMid);
  }
  else
  {
    ll sign = ((drpsv + mqi) % 2) == 0 ? 1 : -1;
    return ((mqi + 1) / 2) * sign;
  }
}

ll Predictor::decalc_pr(ll t, ll qi, ll mev)
{
  if (t == 0)
    return qi;
  return sgn(qi) * std::abs(qi) * (2 * mev + 1);
}

ll Predictor::decalc_sample(ll pr, ll psv)
{
  ll sMin = image_constants.attr("lower_sample_limit").cast<ll>();
  ll sMax = image_constants.attr("upper_sample_limit").cast<ll>();
  return std::clamp(pr + psv, sMin, sMax);
}
