#pragma once
#include "ActivationFunction.hpp"

namespace mlask{

/** @brief class reprsenting Hyperbolic Tangent activation function */
class Tanh: ActivationFunction{
    vectorIn input_;
    float_t activate(float_t input)override;
    float_t derived(float_t input)override;
public:
    /** @brief Construct a new Tanh object
     * @param in Number of inputs to the layer, default is 0, meaning it can be any size*/
    Tanh(std::size_t in = 0);

    /** @brief Try to convert the layer to ONNX format
     * @param graph ONNX graph to which the layer should be added
     * @param input Name of the input tensor
     * @param output Name of the output tensor
     * @return True if the conversion was successful, false otherwise */
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;
    std::string str() const override{ return "Hyperbolic Tangent"; }
};

}
