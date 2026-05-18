#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ostream>
#include "types.hpp"

namespace mlask {

class Layer {
  protected:
    using vectorIn_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using vectorOut_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;

  public:
    virtual ~Layer() = default;
    virtual vectorOut_ forward(vectorIn_) = 0;
    virtual vectorIn_ backward(vectorOut_) = 0;
    virtual void fit(float_t learning_rate) = 0;
    virtual std::ostream& print(std::ostream& os)const{return os;}
};
} // namespace mlask
