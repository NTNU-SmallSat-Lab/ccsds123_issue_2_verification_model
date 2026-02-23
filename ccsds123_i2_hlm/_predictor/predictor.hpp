#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

/*
 * A note on datatypes
 *
 * The algorithm supports up to 32 bits of image data bit depth, so I used the 'long' datatype which guaranties 4 bytes.
 * However, I forgot about unsigned 32 bit, so I needed to switch to 'long long' which uses a lot more data memory.
 * For later I should use a type template to specify the data type when instantiating the class
 */

// #define DEBUG 1 // adds significant execution time

namespace py = pybind11;

template <typename T>
using NumpyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename data_t, ssize_t dims_t>
using _NumpyArr = py::detail::unchecked_mutable_reference<data_t, dims_t>;

class Predictor
{
public:
  Predictor(py::object header, py::object image_constants, NumpyArr<long long> image_sample, bool save_intermediates = false);
  ~Predictor();
  NumpyArr<long long> compress();
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
      std::fill_n(arr.mutable_data(), arr.size(), value);
    }

    template <typename... Args>
    data_t sample(data_t t, Args &&...pos)
    {
      if (enable_sampling)
        arr.template mutable_unchecked<dims_t>()(std::forward<Args>(pos)...) = t;

#ifdef DEBUG
      if (!reference.nbytes()) // this check adds quite a lot of time
        return t;

      if (t == reference.template unchecked<dims_t>()(std::forward<Args>(pos)...))
        return t;

      std::cout << "\n'" << reference_file << "' mismatch position: ";
      ((std::cout << pos << " "), ...);
      std::cout << std::endl;
      throw std::runtime_error("result mismatch");
#endif // DEBUG
      return t;
    }

#ifdef DEBUG
    // allows easy identification of coordinates of deviating values compared to reference
    void set_reference(std::string file_path)
    {
      if (!enable_sampling)
        return;

      py::module_ np = py::module_::import("numpy");
      py::object loaded = np.attr("loadtxt")(file_path, py::arg("delimiter") = std::string(1, ','), py::arg("dtype") = py::dtype::of<data_t>());

      py::tuple shape(dims_t);
      for (ssize_t i = 0; i < dims_t; ++i)
        shape[i] = arr.shape(i);
      py::object reshaped = loaded.attr("reshape")(shape);

      reference = reshaped.cast<NumpyArr<data_t>>();
      reference_file = file_path;
    }
#endif // DEBUG

    NumpyArr<data_t> get_arr() { return arr; }

    template <typename... Args>
    decltype(auto) operator()(Args &&...pos) { return arr.mutable_at(std::forward<Args>(pos)...); }

    bool enable_sampling;

  private:
    NumpyArr<data_t> arr;
    NumpyArr<data_t> reference;

#ifdef DEBUG
    std::string reference_file;
#endif // DEBUG
  };

  /******************** From constructor ********************/

  py::object header;
  py::object image_constants;
  NumpyArr<long long> image_sample; // data cube

  /******************** Samplers ********************/

  // samplers used for intermediate values that can optionally be stored and exported
  std::unique_ptr<Sampler<long long, 3>> lssmpl;    // local sums
  std::unique_ptr<Sampler<long long, 3>> pcdsmpl;   // predicted central local difference
  std::unique_ptr<Sampler<long long, 3>> hrpsvsmpl; // high resolution predictied sample value
  std::unique_ptr<Sampler<long long, 3>> drpsvsmpl; // long long resolution predicted sample value
  std::unique_ptr<Sampler<long long, 3>> psvsmpl;   // predicted sample value
  std::unique_ptr<Sampler<long long, 3>> prsmpl;    // prediction residual
  std::unique_ptr<Sampler<long long, 3>> mevsmpl;   // maximum error value
  std::unique_ptr<Sampler<long long, 3>> qismpl;    // quantizer index
  std::unique_ptr<Sampler<long long, 3>> cqbcsmpl;  // clipped quantizer bin center
  std::unique_ptr<Sampler<long long, 3>> drsrsmpl;  // long long resolution sample representative
  std::unique_ptr<Sampler<long long, 3>> drpesmpl;  // long long resolution prediction error
  std::unique_ptr<Sampler<long long, 3>> tsmpl;     // scaled prediction endpoint difference (theta)

  // these are samplers that need to store their values no matter what
  std::unique_ptr<Sampler<long long, 3>> mqismpl; // mapped quantizer indices
  std::unique_ptr<Sampler<long long, 3>> srsmpl;  // sample representative
  std::unique_ptr<Sampler<long long, 4>> ldvsmpl; // local difference vectors
  std::unique_ptr<Sampler<long long, 4>> wvsmpl;  // weight vectors

  /******************** Constants ********************/

  long long x_size, y_size, z_size;
  bool save_intermediates;

  long long local_difference_values_num;

  long long weight_component_resolution;                         // Symbol: Omega
  long long weight_update_change_interval;                       // Symbol: t_inc
  long long weight_update_initial_parameter;                     // Symbol: nu_min
  long long weight_update_final_parameter;                       // Symbol: nu_max
  long long weight_min;                                          // Symbol: omega_min
  long long weight_max;                                          // Symbol: omega_max
  std::unique_ptr<Sampler<long long, 2>> weight_exponent_offset; // Symbol: Sigma

  long long register_size; // Symbol: R

  std::unique_ptr<Sampler<long long, 2>> absolute_error_limits; // Symbol: a_z
  std::unique_ptr<Sampler<long long, 2>> relative_error_limits; // Symbol: r_z

  /******************** Private methods ********************/

  void init_predictor_constants();
  void init_predictor_arrays();
  void init_weights();

  long long calc_local_sum(long long x, long long y, long long z);
  std::vector<long long> calc_local_difference_vector(long long x, long long y, long long z, long long local_sum, long long prev_local_sum);
  long long calc_predicted_central_local_diff(long long x, long long y, long long z);
  long long calc_high_resolution_pred_sample_value(long long x, long long y, long long z, long long local_sum, long long predicted_central_local_diff);
  long long calc_double_resolution_predicted_sample_value(long long x, long long y, long long z, long long high_resolution_pred_sample_value);
  long long calc_predicted_sample_value(long long double_resolution_predicted_sample_value);
  long long calc_prediction_residual(long long sample, long long predicted_sample_value);
  long long calc_maximum_error(long long y, long long z, long long predicted_sample_value);
  long long calc_quantizer_index(long long t, long long maximum_error, long long prediction_residual);
  long long calc_clipped_quantizer_bin_center(long long x, long long y, long long z, long long predicted_sample_value, long long maximum_error, long long quantizer_index);
  long long calc_double_resolution_sample_representative(long long z, long long clipped_quantizer_bin_center, long long quantizer_index, long long maximum_error, long long high_resolution_pred_sample_value);
  long long calc_sample_representative(long long x, long long y, long long z, long long clipped_quantizer_bin_center, long long double_resolution_sample_representative);
  long long calc_double_resolution_prediction_error(long long clipped_quantizer_bin_center, long long double_resolution_predicted_sample_value);
  long long calc_theta(long long t, long long predicted_sample_value, long long maximum_error);
  long long calc_mapped_quantizer_index(long long quantizer_index, long long theta, long long double_resolution_predicted_sample_value);
  std::vector<long long> calc_weight_vector(long long x, long long y, long long z, long long double_resolution_prediction_error);
};
