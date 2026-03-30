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
 * However, I forgot about unsigned 32 bit, so I needed to switch to 'long long' () which uses a lot more data memory.
 * For later I should use a type template to specify the data type when instantiating the class
 */

// #define DEBUG 1 // adds significant execution time

namespace py = pybind11;

template <typename T>
using NumpyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename data_t, ssize_t dims_t>
using _NumpyArr = py::detail::unchecked_mutable_reference<data_t, dims_t>;

typedef long long ll;

// forward declaration
template <typename data_t, ssize_t dims_t>
class Sampler;

class Predictor
{
public:
  Predictor(py::object header, py::object image_constants, bool delayed_weight_updates = false, bool save_intermediates = false);
  ~Predictor();
  NumpyArr<ll> compress(NumpyArr<ll> image_sample);
  NumpyArr<ll> decompress(NumpyArr<ll> mqi);
  void save_data(std::string output_folder);

private:
  /******************** From constructor ********************/

  py::object header;
  py::object image_constants;

  /******************** Samplers ********************/

  // samplers used for intermediate values that can optionally be stored and exported
  std::unique_ptr<Sampler<ll, 3>> lssmpl;    // local sums
  std::unique_ptr<Sampler<ll, 3>> pcdsmpl;   // predicted central local difference
  std::unique_ptr<Sampler<ll, 3>> hrpsvsmpl; // high resolution predictied sample value
  std::unique_ptr<Sampler<ll, 3>> drpsvsmpl; // double resolution predicted sample value
  std::unique_ptr<Sampler<ll, 3>> psvsmpl;   // predicted sample value
  std::unique_ptr<Sampler<ll, 3>> prsmpl;    // prediction residual
  std::unique_ptr<Sampler<ll, 3>> mevsmpl;   // maximum error value
  std::unique_ptr<Sampler<ll, 3>> qismpl;    // quantizer index
  std::unique_ptr<Sampler<ll, 3>> cqbcsmpl;  // clipped quantizer bin center
  std::unique_ptr<Sampler<ll, 3>> drsrsmpl;  // double resolution sample representative
  std::unique_ptr<Sampler<ll, 3>> tsmpl;     // scaled prediction endpoint difference (theta)

  // these are samplers that need to store their values no matter what
  std::unique_ptr<Sampler<ll, 3>> mqismpl;  // mapped quantizer indices
  std::unique_ptr<Sampler<ll, 3>> srsmpl;   // sample representative
  std::unique_ptr<Sampler<ll, 3>> drpesmpl; // double resolution prediction error
  std::unique_ptr<Sampler<ll, 4>> ldvsmpl;  // local difference vectors
  std::unique_ptr<Sampler<ll, 4>> wvsmpl;   // weight vectors

  /******************** Constants ********************/

  ll x_size, y_size, z_size;

  bool save_intermediates;
  bool delayed_weight_updates;

  ll local_difference_values_num;

  ll weight_component_resolution;                         // Symbol: Omega
  ll weight_update_change_interval;                       // Symbol: t_inc
  ll weight_update_initial_parameter;                     // Symbol: nu_min
  ll weight_update_final_parameter;                       // Symbol: nu_max
  ll weight_min;                                          // Symbol: omega_min
  ll weight_max;                                          // Symbol: omega_max
  std::unique_ptr<Sampler<ll, 2>> weight_exponent_offset; // Symbol: Sigma

  ll register_size; // Symbol: R

  std::unique_ptr<Sampler<ll, 2>> absolute_error_limits; // Symbol: a_z
  std::unique_ptr<Sampler<ll, 2>> relative_error_limits; // Symbol: r_z

  /******************** Private methods ********************/

  void init_predictor_constants();
  void init_predictor_arrays();
  void init_weights();

  ll calc_ls(ll x, ll y, ll z);
  std::vector<ll> calc_ldv(ll x, ll y, ll z, ll ls, ll prev_ls);
  ll calc_pcd(ll x, ll y, ll z);
  ll calc_hrpsv(ll x, ll y, ll z, ll ls, ll pcd);
  ll calc_drpsv(ll x, ll y, ll z, ll hrpsv, NumpyArr<ll> image_sample);
  ll calc_psv(ll drpsv);
  ll calc_pr(ll sample, ll psv);
  ll calc_mev(ll y, ll z, ll psv);
  ll calc_qi(ll t, ll mev, ll pr);
  ll calc_cqbc(ll x, ll y, ll z, ll psv, ll mev, ll qi, NumpyArr<ll> image_sample);
  ll calc_drsr(ll z, ll cqbc, ll qi, ll mev, ll hrpsv);
  ll calc_sample_representative(ll x, ll y, ll z, ll cqbc, ll drsr, NumpyArr<ll> image_sample);
  ll calc_drpe(ll cqbc, ll drpsv);
  ll calc_theta(ll t, ll psv, ll mev);
  ll calc_mqi(ll qi, ll theta, ll drpsv);
  std::vector<ll> calc_weight_vector(ll x, ll y, ll z);

  ll decalc_qi(ll theta, ll mqi, ll psv, ll drpsv);
  ll decalc_pr(ll t, ll qi, ll mev);
  ll decalc_sample(ll pr, ll psv);
};

/******************** Sampler utility class ********************/

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
