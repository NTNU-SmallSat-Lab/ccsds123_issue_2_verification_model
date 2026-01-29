#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T>
using NumpyArr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
using unchecked3D_NumpyArr = py::detail::unchecked_reference<T, 3>;

class Predictor {
  public:
    Predictor(py::object header, py::object image_constants, NumpyArr<int> image_sample);
    NumpyArr<int> compress(); 

  private:
    /******************** Sampler utility ********************/

    template <typename T>
    class Sampler {
      public:
        Sampler(std::array<ssize_t,3> dimensions, bool enable_sampling = false) 
          : enable_sampling(enable_sampling)
        {
          if (!enable_sampling) return;
          // allocate sampler array
          arr = NumpyArr<T>(dimensions);
          _arr.emplace(arr.template unchecked<3>());
        };
        T sample(T t, int x, int y, int z) {
          if (enable_sampling)
            _arr(x, y, z) = t;
          return t;
        };

      private:
        bool enable_sampling;
        NumpyArr<T>                              arr;
        std::optional<unchecked3D_NumpyArr<T>>  _arr;
    };

    /******************** Read only arrays ********************/

    py::object header;
    py::object image_constants;
    unchecked3D_NumpyArr<int> _image_sample;           // three dimensional image cube

    /******************** Samplers ********************/
    // samplers are used for intermediate values that can optionally be stored and exported

    Sampler<int> *lssmpl;
    
    /******************** Constants ********************/

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
};
