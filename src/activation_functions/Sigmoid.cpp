#include "Sigmoid.hpp"
#include <cmath>

namespace mlask{
Sigmoid::Sigmoid(std::size_t in){
    in_ = in;
    out_ = in;
}

float_t Sigmoid::activate(float_t input){
   return 1/(1 + exp(-input));
}

float_t Sigmoid::derived(float_t input){
   return activate(input) * (1 - activate(input));
}

bool Sigmoid::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const{
    onnx::NodeProto* relu_node = graph->add_node();
    relu_node->set_op_type("Sigmoid");
    relu_node->add_input(input);
    relu_node->add_output(output);
    return true;
}

}
