#include "predictor.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_predictor, m, py::mod_gil_not_used())
{
  py::class_<Predictor>(m, "Predictor")
      .def(py::init<py::object, py::object, NumpyArr<long long>, bool>())
      .def("compress", &Predictor::compress)
      .def("save_data", &Predictor::save_data);
}
