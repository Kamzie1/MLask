#include "Model.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "ActivationFunction.hpp"
#include "InternalActivationFunction.hpp"
#include "Layer.hpp"
#include "Relu.hpp"

namespace mlask {
Model::Model(std::size_t size): layers_(size) {}
Model::Model(std::size_t epochs, bool log): epochs_(epochs), log_(log) {}
Model::Model(std::size_t epochs, bool log): epochs_(epochs), log_(log) {}

void Model::addLayer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    LOG("Added Layer");
}

vectorOut_ Model::forward(vectorIn_ input) const {
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    return input;
}

void Model::backprop(vectorIn_ input, vectorOut_ expected, err_function err){
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    vectorOut_ error = err(input, expected);
    for (auto it = layers_.rbegin(); it!=layers_.rend(); ++it){
        error = (*it).get()->backward(error);
    }
}
void Model::fit(float_t learning_rate){
    for (const std::unique_ptr<Layer> &layer : layers_) {
        layer->fit(learning_rate);
    }
}


void Model::addActivationFunctionWithLambdas( actfunc func, actfunc derv){
    layers_.push_back(std::make_unique<ActivationFunction>(func, derv));
    LOG("Added Activation Function");
}

void Model::addActivationFunction(InternalActivationFunction activationFunction){
    switch(activationFunction){
        #define REGISTER(X)\
            case InternalActivationFunction::X:\
            {\
                layers_.push_back(std::make_unique<X>());\
                std::cout<<"[INFO] Added "<< #X<<std::endl;\
                break;\
            }
            LIST_OF_ACTIVATION_FUNCTIONS
        #undef REGISTER
        default:
            throw std::invalid_argument("There is no such layer");
    }
}
} // namespace mlask
