#include "predictor.hpp"
#include <iostream>

Predictor::Predictor(py::object header) : header(header){
  std::cout << "Predictor constructor" << std::endl;

  std::cout << "Ogus: " << header.attr("unary_length_limit").cast<int>() << std::endl;
}

void Predictor::runPredictor() {
  std::cout << "Running predictor mogus!!" << std::endl;
}
