#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include "types.hpp"
#include "onnx/onnx_pb.h"

namespace mlask {

/** @brief Base class for all layers in the neural network */
class Layer {
protected:
    std::size_t in_;
    std::size_t out_;
  public:
    std::size_t getIn()const noexcept{ return in_; }
    std::size_t getOut()const noexcept{ return out_; }

    virtual ~Layer() = default;
    /** @brief Defines a way to move forward in a neural network */
    virtual vectorOut forward(vectorIn) = 0;
    /** @brief Defines a way to backpropagate error in backropagation algorithm */
    virtual vectorIn backward(vectorOut) = 0;
    /** @brief Describes how the layer 'learns', meaning it defines how layer updates itself. */
    virtual void fit(float_t learning_rate) = 0;
    /** @brief Converts the layer to ONNX format. If the layer cannot be converted, it should return false. */
    virtual bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output)const { return false; }
    /** @brief Returns a string representation of the layer. */
    [[nodiscard]] virtual std::string str()const{ return "Layer"; }
};

/** @brief concept ensuring a class derives from Layer*/
template <typename T>
concept TLayer= std::derived_from<T, Layer>;
} // namespace mlask
