#pragma once
#include "Layer.hpp"
#include "Eigen/Dense"

namespace mlask{
class ActivationFunction : public Layer{
    using function = std::function<float_t(float_t)>;

    function func_;
    function derv_;

    Eigen::Matrix<float_t, Eigen::Dynamic, 1> input_;
public:
    ActivationFunction(function func, function derv):func_(std::move(func)), derv_(std::move(derv)){}
    vectorOut_ forward(vectorIn_ input) override;
    vectorIn_ backward(vectorOut_ error) override;
    void fit(float_t learning_rate) override {}
};
}
