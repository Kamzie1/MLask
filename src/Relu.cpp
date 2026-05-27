#include <cmath>
#include "Relu.hpp"

namespace mlask{
Relu::Relu(std::size_t in){
    in_ = in;
    out_ = in;
}

vectorOut_ Relu::forward(vectorIn_ input){
    input_ = input;
    return input.unaryExpr([](float_t x){ return std::max(0.f,x); });
}

vectorIn_ Relu::backward(vectorOut_ error){
    return input_.unaryExpr([](float_t x){ return (x > 0.0f) ? 1.0f : 0.0f; }).transpose() * error;
}

bool Relu::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const{
    onnx::NodeProto* relu_node = graph->add_node();
    relu_node->set_op_type("Relu");
    relu_node->add_input(input);
    relu_node->add_output(output);
    return true;
}

}
