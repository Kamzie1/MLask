#include "SoftMax.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace mlask{
vectorOut SoftMax::forward(vectorIn input) {
    input_ = input;
    float_t sum = input.unaryExpr([this](float_t x){ return exp(x); }).sum();
    forwardOutput_ = input.unaryExpr([this, sum](float_t x){ return exp(x) / sum; });
    return forwardOutput_;
}

vectorIn SoftMax::backward(vectorOut error) {
    auto s = forwardOutput_;

    float_t s_dot_error = s.dot(error); 
    vectorIn gradient = s.cwiseProduct(error) - (s * s_dot_error);

    return gradient;
}
}

