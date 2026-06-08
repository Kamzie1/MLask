#include <cmath>
#include "Relu.hpp"

namespace mlask{
Relu::Relu(std::size_t in){
    in_ = in;
    out_ = in;
}

float_t Relu::activate(float_t input){
   return std::max<float_t>(0,input); 
}

float_t Relu::derived(float_t input){
   return (input < 0) ? 0 : 1;
}

bool Relu::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const{
    onnx::NodeProto* relu_node = graph->add_node();
    relu_node->set_op_type("Relu");
    relu_node->add_input(input);
    relu_node->add_output(output);
    return true;
}

}
