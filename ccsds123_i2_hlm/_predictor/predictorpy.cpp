#include <pybind11/pybind11.h>
#include "predictor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_predictor, m, py::mod_gil_not_used()) {
    py::class_<Predictor>(m, "Predictor")
        .def(py::init<py::object>())
        .def("runPredictor", &Predictor::runPredictor);
}
