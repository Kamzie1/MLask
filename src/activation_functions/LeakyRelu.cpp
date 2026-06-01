#include <cmath>
#include "LeakyRelu.hpp"

namespace mlask{
LeakyRelu::LeakyRelu(float_t a, std::size_t in): a_(a){
    in_ = in;
    out_ = in;
}

float_t LeakyRelu::activate(float_t input){
   return (input < 0) ? a_ * input : input;
}

float_t LeakyRelu::derived(float_t input){
   return (input < 0) ? a_ : 1;
}

bool LeakyRelu::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const{
    onnx::NodeProto* relu_node = graph->add_node();
    relu_node->set_op_type("LeakyRelu");
    relu_node->add_input(input);
    relu_node->add_output(output);
    return true;
}

}
