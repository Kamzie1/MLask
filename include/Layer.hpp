#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include "types.hpp"

namespace mlask {

class Layer {
  protected:
    using vectorIn_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using vectorOut_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;

  public:
    virtual ~Layer() = default;
    virtual vectorOut_ forward(vectorIn_) const = 0;
    virtual vectorOut_ backward(vectorIn_) = 0;
    virtual void fit() = 0;
};
} // namespace mlask
