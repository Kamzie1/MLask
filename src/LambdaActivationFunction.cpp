#include "LambdaActivationFunction.hpp"
#include <Eigen/Dense>

namespace mlask{
vectorOut LambdaActivationFunction::forward(vectorIn input) {
    input_ = input;
    return input.unaryExpr([this](float_t x){ return func_(x); });
}

vectorIn LambdaActivationFunction::backward(vectorOut error) {
    auto derivative =  input_.unaryExpr([this](float_t x){ return derv_(x); });
    return error.cwiseProduct(derivative);
}
}
