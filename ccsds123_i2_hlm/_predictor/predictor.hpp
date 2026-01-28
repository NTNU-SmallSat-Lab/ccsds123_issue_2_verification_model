#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

class Predictor {
  public:
    Predictor(py::object header);
    void runPredictor(); 

  private:
    py::object header; // image header file
};
