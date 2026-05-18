#include <cmath>
#include "Relu.hpp"

namespace mlask{
Relu::vectorOut_ Relu::forward(vectorIn_ input){
    input_ = input;
    return input.unaryExpr([](float_t x){ return std::max(0.f,x); });
}

Relu::vectorIn_ Relu::backward(vectorOut_ error){
    return input_.unaryExpr([](float_t x){ return (x > 0.0f) ? 1.0f : 0.0f; }).transpose() * error;
}
}
