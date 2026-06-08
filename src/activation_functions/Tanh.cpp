
#include "Tanh.hpp"
#include <cmath>

namespace mlask{
Tanh::Tanh(std::size_t in){
    in_ = in;
    out_ = in;
}

float_t Tanh::activate(float_t input){
   return (float_t)(exp(input) - exp(-input))/(exp(input) + exp(-input));
}

float_t Tanh::derived(float_t input){
   return 1 - (activate(input) * activate(input));
}

bool Tanh::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const{
    onnx::NodeProto* relu_node = graph->add_node();
    relu_node->set_op_type("Tanh");
    relu_node->add_input(input);
    relu_node->add_output(output);
    return true;
}

}
