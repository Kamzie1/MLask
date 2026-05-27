#pragma once
#include "Layer.hpp"
#include <Eigen/Core>

namespace mlask{
/** @brief Layer representing Relu activation function */
class Relu : public Layer{
    vectorIn_ input_;
public:
    /** @brief Construct a new Relu object
     * @param in Number of inputs to the layer, default is 0, meaning it can be any size*/
    Relu(std::size_t in = 0);
    /** @brief Forward pass through the layer
     * @param input Input vector to the layer
     * @return Output vector after applying Relu activation function */
    vectorOut_ forward(vectorIn_ input) override;
    /** @brief Backward pass through the layer
     * @param error Gradient of the loss with respect to the output of the layer
     * @return Gradient of the loss with respect to the input of the layer */
    vectorIn_ backward(vectorOut_ error) override;
    /** @brief Fit the layer to the data, not applicable for Relu layer */
    void fit(float_t learning_rate) override {}
    /** @brief Try to convert the layer to ONNX format
     * @param graph ONNX graph to which the layer should be added
     * @param input Name of the input tensor
     * @param output Name of the output tensor
     * @return True if the conversion was successful, false otherwise */
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;
    std::string cstr() const override{
        return "Relu";
    }
};
}
