#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ostream>
#include <stdexcept>
#include "types.hpp"
#include "onnx/onnx_pb.h"

namespace mlask {

class Layer {
protected:
    std::size_t in_;
    std::size_t out_;
  public:
    std::size_t getIn()const{ return in_; }
    std::size_t getOut()const{ return out_; }

    virtual ~Layer() = default;
    ///@brief Defines a way to move forward in a neural network
    virtual vectorOut_ forward(vectorIn_) = 0;
    ///@brief Defines a way to backpropagate error in backropagation algorithm
    virtual vectorIn_ backward(vectorOut_) = 0;
    ///@brief Describes how the layer 'learns', meaning it defines how layer updates itself.
    virtual void fit(float_t learning_rate) = 0;
    ///@brief Converts the layer to ONNX format. If the layer cannot be converted, it should return false.
    virtual bool tryConvertToONNX(onnx::GraphProto* graph)const { return false; }
    virtual std::ostream& print(std::ostream& os)const{return os;}
};
} // namespace mlask
