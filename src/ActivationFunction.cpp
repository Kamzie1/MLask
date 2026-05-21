//#include "ActivationFunction.hpp"
//#include <Eigen/Dense>
//
//namespace mlask{
//vectorOut_ ActivationFunction::forward(vectorIn_ input) {
//    input_ = input;
//    return input.unaryExpr([this](float_t x){ return func_(x); });
//}
//
//vectorIn_ ActivationFunction::backward(vectorOut_ error) {
//    auto derivative =  input_.unaryExpr([this](float_t x){ return derv_(x); });
//    return error.cwiseProduct(derivative);
//}
//}
