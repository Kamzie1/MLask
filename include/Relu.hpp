#pragma once
#include "Layer.hpp"
#include <Eigen/Core>

namespace mlask{
/** @brief Layer representing Relu activation function */
class Relu : public Layer{
    vectorIn_ input_;
public:
    Relu(std::size_t in = 0);
    vectorOut_ forward(vectorIn_ input) override;
    vectorIn_ backward(vectorOut_ error) override;
    void fit(float_t learning_rate) override {}
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;
};
}
