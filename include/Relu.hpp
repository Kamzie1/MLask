#pragma once
#include "Layer.hpp"
#include <Eigen/Core>

namespace mlask{
class Relu : public Layer{
    Eigen::Matrix<float_t, Eigen::Dynamic, 1> input_;
public:
    vectorOut_ forward(vectorIn_ input) override;
    vectorIn_ backward(vectorOut_ error) override;
    void fit(float_t learning_rate) override {}
};
}
