#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

template <typename T>
using NumpyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename data_t, ssize_t dims_t>
using _NumpyArr = py::detail::unchecked_mutable_reference<data_t, dims_t>;

class Predictor {
  public:
    Predictor(py::object header, py::object image_constants, NumpyArr<int> image_sample);
    NumpyArr<int> compress(); 
    void save_data(std::string output_folder);

  private:
    /******************** Sampler utility ********************/

    template <typename data_t, ssize_t dims_t>
    class Sampler {
      public:
        Sampler(std::array<ssize_t, dims_t> dimensions, bool enable_sampling = true) 
          : enable_sampling(enable_sampling)
        {
          if (!enable_sampling) return;
          // allocate sampler array
          arr = NumpyArr<data_t>(dimensions);
          _arr.emplace(arr.template mutable_unchecked<dims_t>());
        }

        template <typename... Args>
        data_t sample(data_t t, Args &&...pos) {
          if (enable_sampling)
            (*_arr)(std::forward<Args>(pos)...) = t;
          return t;
        }

        template <typename... Args>
        data_t operator()(Args &&...pos) { return (*_arr)(std::forward<Args>(pos)...); }

        NumpyArr<data_t> get_arr() { return arr; }

        bool enable_sampling;

      private:
        NumpyArr<data_t>                          arr;
        std::optional<_NumpyArr<data_t, dims_t>> _arr;
    };  

    /******************** Read only arrays ********************/

    py::object header;
    py::object image_constants;
    _NumpyArr<int, 3> _image_sample; // data cube

    /******************** Samplers ********************/

    // samplers used for intermediate values that can optionally be stored and exported
    Sampler<int, 3> *lssmpl; // local sums

    // these are samplers that need to store their values no matter what
    Sampler<int, 3> *mqismpl; // mapped quantizer indices
    Sampler<int, 3> *repsmpl; // sample representatives
    Sampler<int, 4> *ldvsmpl; // local difference vectors
   
    /******************** Constants ********************/

    int x_size, y_size, z_size;

    int local_difference_values_num;

    int weight_component_resolution;     // Symbol: Omega
    int weight_update_change_interval;   // Symbol: t_inc
    int weight_update_initial_parameter; // Symbol: nu_min
    int weight_update_final_parameter;   // Symbol: nu_max
    int weight_exponent_offset;          // Symbol: sigma (in word-final position)
    int weight_min;                      // Symbol: omega_min
    int weight_max;                      // Symbol: omega_max

    int register_size; // Symbol: R

    // for full support of standard one should use vectors here
    int absolute_error_limit; // Symbol: a_z
    int relative_error_limit; // Symbol: r_z
    // std::vector<int> absolute_error_limits; // Symbol: a_z
    // std::vector<int> relative_error_limits; // Symbol: r_z

    /******************** Private methods ********************/
  
    void init_predictor_constants();
    void init_predictor_arrays();

    int calc_local_sum(int x, int y, int z, Sampler<int, 3> *repsmpl);
    std::vector<int> calc_local_difference_vector(int x, int y, int z, int local_sum, int prev_local_sum,
                                                  Sampler<int, 3> *repsmpl, Sampler<int, 4> *ldvsmpl);
};
