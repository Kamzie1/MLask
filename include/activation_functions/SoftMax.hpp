#pragma once
#include "Layer.hpp"

namespace mlask{

/** @brief class reprsenting Soft Max activation function, since the definition is to complex for activation function it derives from Layer */
class SoftMax: Layer{
    vectorIn input_;
    vectorOut forwardOutput_;
public:
    /** @brief Construct a new SoftMax object
     * @param in Number of inputs to the layer, default is 0, meaning it can be any size*/
    SoftMax(std::size_t in = 0);

    /** @brief softmax definition */
    vectorOut forward(vectorIn)override;
    /** @brief derived version of softmax */
    vectorIn backward(vectorOut)override;
    /** @brief This does nothing, since softmax doesnt change. */
    void fit(float_t learning_rate)override{}

    /** @brief Try to convert the layer to ONNX format
     * @param graph ONNX graph to which the layer should be added
     * @param input Name of the input tensor
     * @param output Name of the output tensor
     * @return True if the conversion was successful, false otherwise */
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;
    std::string str() const override{ return "SoftMax"; }
};

}
