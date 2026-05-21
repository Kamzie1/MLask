#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ostream>
#include "types.hpp"

namespace mlask {

class Layer {
  public:
    virtual ~Layer() = default;
    // Defines a way to move forward in a neural network
    virtual vectorOut_ forward(vectorIn_) = 0;
    // Defines a way to backpropagate error in backropagation algorithm
    virtual vectorIn_ backward(vectorOut_) = 0;
    // This is when the layer 'learns', meaning it updates itself.
    virtual void fit(float_t learning_rate) = 0;
    virtual std::ostream& print(std::ostream& os)const{return os;}
};
} // namespace mlask
