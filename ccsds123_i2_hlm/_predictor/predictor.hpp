#pragma once

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename T>
using NumpyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename data_t, ssize_t dims_t>
using _NumpyArr = py::detail::unchecked_mutable_reference<data_t, dims_t>;

class Predictor
{
public:
  Predictor(py::object header, py::object image_constants, NumpyArr<long> image_sample, bool save_intermediates = false);
  NumpyArr<long> compress();
  void save_data(std::string output_folder);

private:
  /******************** Sampler utility ********************/

  template <typename data_t, ssize_t dims_t>
  class Sampler
  {
  public:
    Sampler(std::array<ssize_t, dims_t> dimensions, data_t value, bool enable_sampling = true)
        : enable_sampling(enable_sampling)
    {
      if (!enable_sampling)
        return;
      // allocate sampler array
      arr = NumpyArr<data_t>(dimensions);
      std::memset(arr.mutable_data(), value, arr.nbytes());
      _arr.emplace(arr.template mutable_unchecked<dims_t>());
    }

    template <typename... Args>
    data_t sample(data_t t, Args &&...pos)
    {
      if (enable_sampling)
        (*_arr)(std::forward<Args>(pos)...) = t;
      return t;
    }

    NumpyArr<data_t> get_arr() { return arr; }

    template <typename... Args>
    decltype(auto) operator()(Args &&...pos) { return (*_arr)(std::forward<Args>(pos)...); }

    bool enable_sampling;

  private:
    NumpyArr<data_t> arr;
    std::optional<_NumpyArr<data_t, dims_t>> _arr;
  };

  /******************** Read only arrays ********************/

  py::object header;
  py::object image_constants;
  _NumpyArr<long, 3> _image_sample; // data cube

  /******************** Samplers ********************/

  // samplers used for longermediate values that can optionally be stored and exported
  Sampler<long, 3> *lssmpl;    // local sums
  Sampler<long, 3> *pcdsmpl;   // predicted central local difference
  Sampler<long, 3> *hrpsvsmpl; // high resolution predictied sample value
  Sampler<long, 3> *drpsvsmpl; // double resolution predicted sample value
  Sampler<long, 3> *psvsmpl;   // predicted sample value
  Sampler<long, 3> *prsmpl;    // prediction residual
  Sampler<long, 3> *mevsmpl;   // maximum error value
  Sampler<long, 3> *qismpl;    // quantizer index
  Sampler<long, 3> *cqbcsmpl;  // clipped quantizer bin center
  Sampler<long, 3> *drsrsmpl;  // double resolution sample representative
  Sampler<long, 3> *srsmpl;    // sample representative
  Sampler<long, 3> *drpesmpl;  // double resolution prediction error
  Sampler<long, 3> *tsmpl;     // scaled prediction endpoint difference (theta)

  // these are samplers that need to store their values no matter what
  Sampler<long, 3> *mqismpl; // mapped quantizer indices
  Sampler<long, 3> *repsmpl; // sample representatives
  Sampler<long, 4> *ldvsmpl; // local difference vectors
  Sampler<long, 4> *wvsmpl;  // weight vectors

  /******************** Constants ********************/

  long x_size, y_size, z_size;
  bool save_intermediates;

  long local_difference_values_num;

  long weight_component_resolution;         // Symbol: Omega
  long weight_update_change_interval;       // Symbol: t_inc
  long weight_update_initial_parameter;     // Symbol: nu_min
  long weight_update_final_parameter;       // Symbol: nu_max
  long weight_min;                          // Symbol: omega_min
  long weight_max;                          // Symbol: omega_max
  Sampler<long, 2> *weight_exponent_offset; // Symbol: Sigma

  long register_size; // Symbol: R

  Sampler<long, 2> *absolute_error_limits; // Symbol: a_z
  Sampler<long, 2> *relative_error_limits; // Symbol: r_z

  /******************** Private methods ********************/

  void init_predictor_constants();
  void init_predictor_arrays();
  void init_weights();

  long calc_local_sum(long x, long y, long z);
  std::vector<long> calc_local_difference_vector(long x, long y, long z, long local_sum, long prev_local_sum);
  long calc_predicted_central_local_diff(long x, long y, long z);
  long calc_high_resolution_pred_sample_value(long x, long y, long z, long local_sum, long predicted_central_local_diff);
  long calc_double_resolution_predicted_sample_value(long x, long y, long z, long high_resolution_pred_sample_value);
  long calc_predicted_sample_value(long double_resolution_predicted_sample_value);
  long calc_prediction_residual(long sample, long predicted_sample_value);
  long calc_maximum_error(long y, long z, long predicted_sample_value);
  long calc_quantizer_index(long t, long maximum_error, long prediction_residual);
  long calc_clipped_quantizer_bin_center(long x, long y, long z, long predicted_sample_value, long maximum_error, long quantizer_index);
  long calc_double_resolution_sample_representative(long z, long clipped_quantizer_bin_center, long quantizer_index, long maximum_error, long high_resolution_pred_sample_value);
  long calc_sample_representative(long x, long y, long z, long clipped_quantizer_bin_center, long double_resolution_sample_representative);
  long calc_double_resolution_prediction_error(long clipped_quantizer_bin_center, long double_resolution_predicted_sample_value);
  long calc_theta(long t, long predicted_sample_value, long maximum_error);
  long calc_mapped_quantizer_index(long quantizer_index, long theta, long double_resolution_predicted_sample_value);
  std::vector<long> calc_weight_vector(long x, long y, long z, long double_resolution_prediction_error);
};
