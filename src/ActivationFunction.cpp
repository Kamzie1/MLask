#include "ActivationFunction.hpp"
#include <Eigen/Dense>

namespace mlask{
vectorOut ActivationFunction::forward(vectorIn input) {
    input_ = input;
    return input.unaryExpr([this](float_t x){ return activate(x); });
}

vectorIn ActivationFunction::backward(vectorOut error) {
    auto derivative =  input_.unaryExpr([this](float_t x){ return derived(x); });
    return error.cwiseProduct(derivative);
}
}
