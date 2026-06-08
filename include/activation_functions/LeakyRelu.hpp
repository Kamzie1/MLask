#pragma once
#include "ActivationFunction.hpp"
#include <string>

namespace mlask{

/** @brief Class representing Leaky Relu activation function*/
class LeakyRelu : public ActivationFunction{
    float_t a_;
    vectorIn input_;
    float_t activate(float_t input)override;
    float_t derived(float_t input)override;
public:
    /** @brief Construct a new LeakyRelu object
     * @param a an a for leaky relu definition
     * @param in Number of inputs to the layer, default is 0, meaning it can be any size
     * */
    LeakyRelu(float_t a, std::size_t in = 0);

    /** @brief Try to convert the layer to ONNX format
     * @param graph ONNX graph to which the layer should be added
     * @param input Name of the input tensor
     * @param output Name of the output tensor
     * @return True if the conversion was successful, false otherwise */
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;
    std::string str() const override{ return "Leaky Relu with a=" + std::to_string(a_); }
};

}
