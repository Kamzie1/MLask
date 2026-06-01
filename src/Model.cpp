#include "Model.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "GenericErrorFunction.hpp"
#include "LambdaActivationFunction.hpp"
#include "InternalActivationFunction.hpp"
#include "Layer.hpp"
#include "Relu.hpp"
#include "ProgressBar.hpp"
#include "onnx/onnx_pb.h"
#include <fstream>
#include <string>
#include <unistd.h>
#include "config.h"

#define INPUT "X"
#define OUTPUT "Y"

namespace mlask {
Model::Model(std::size_t in, std::size_t out, std::size_t size, std::size_t epochs, bool log):
    in_(in), out_(out), epochs_(epochs), log_(log) {layers_.reserve(size);}

void Model::addLayer(std::unique_ptr<Layer> layer) {
    std::size_t out = lastOut();
    if(layer->getIn() != out && layer->getIn()!=0){
        ERR("Input size of the layer does not match the output size of the previous layer");
        throw std::runtime_error("Input size of the layer does not match the output size of the previous layer");
    }
    else{
        layers_.push_back(std::move(layer));
        LOG("Added Layer");
    }
}

vectorOut Model::forward(vectorIn input) const {
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    return input;
}

void Model::fit(float_t learning_rate){
    epoch_++;
    for (const std::unique_ptr<Layer> &layer : layers_) {
        layer->fit(learning_rate);
    }
    if(log_){
        ProgressBar::draw((float)epoch_ / (float)epochs_);
    }
}

void Model::addLambdaActivationFunction( actfunc func, actfunc derv){
    layers_.push_back(std::make_unique<LambdaActivationFunction>(func, derv, lastOut()));
    LOG("Added Activation Function");
}

void Model::addActivationFunction(InternalActivationFunction activationFunction){
    switch(activationFunction){
        #define REGISTER(X)\
            case InternalActivationFunction::X:\
            {\
                layers_.push_back(std::make_unique<X>(lastOut()));\
                std::cout<<"[INFO] Added "<< #X<<std::endl;\
                break;\
            }
            LIST_OF_ACTIVATION_FUNCTIONS
        #undef REGISTER
        default:
            std::string message = "There is no such internal activation function. See InternalActivationFunction enum.";
            ERR(message);
            throw std::invalid_argument(message);
    }
}

bool Model::tryConvertToONNX(onnx::ModelProto& model, std::string name) const{
    model.set_ir_version(ONNX_IR_VERSION);
    onnx::OperatorSetIdProto* opset = model.add_opset_import();
    opset->set_version(ONNX_OPSET_VERSION);

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name(name);

    std::string input = INPUT;
    std::string output;

    std::size_t index= 0;
    for(auto&& layer : layers_){
        if(index + 1 == layers_.size()){
            output = OUTPUT;
        }
        else{
            output = "hidden_" + std::to_string(index);
        }
        if(!layer->tryConvertToONNX(graph, input, output)){
            ERR("Failed to convert a layer at position " + index + " to ONNX format.")
            return false;
        }
        input = output;
        index++;
    }

    /* input X */
    onnx::ValueInfoProto* input_x = graph->add_input();
    input_x->set_name(INPUT);
    onnx::TypeProto_Tensor* x_tensor_type = input_x->mutable_type()->mutable_tensor_type();
    x_tensor_type->set_elem_type(onnx::TensorProto::FLOAT);
    x_tensor_type->mutable_shape()->add_dim()->set_dim_param("batch_size");
    x_tensor_type->mutable_shape()->add_dim()->set_dim_value(in_);

    /* Output Y */
    onnx::ValueInfoProto* output_y = graph->add_output();
    output_y->set_name(OUTPUT);
    onnx::TypeProto_Tensor* y_tensor_type = output_y->mutable_type()->mutable_tensor_type();
    y_tensor_type->set_elem_type(onnx::TensorProto::FLOAT);
    y_tensor_type->mutable_shape()->add_dim()->set_dim_param("batch_size");
    y_tensor_type->mutable_shape()->add_dim()->set_dim_value(out_);
    return true;
}

void Model::exportToONNX(std::filesystem::path path, std::string name)const{
    onnx::ModelProto model;
    if(!tryConvertToONNX(model, name)){
        ERR("Failed to convert model to ONNX");
        return;
    }
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model.SerializeToOstream(&output)) {
        ERR("Failed to write ONNX file to disk");
    }
    LOG("Succesfully written ONNX file to disk");
}

std::string Model::str()const{
    std::string model_cstr;
    for (const std::unique_ptr<Layer> &layer : layers_) {
        model_cstr = model_cstr +  "\n|\n" + "V\n";
        model_cstr += layer->str();
    }
    return "Model with " + std::to_string(layers_.size()) + " layers:" + model_cstr;
}
} // namespace mlask
