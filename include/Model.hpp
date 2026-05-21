#pragma once
#include "Layer.hpp"
#include "InternalActivationFunction.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <vector>
#include "FullyConnectedLayer.hpp"
#include <iostream>

#ifdef DEBUG

#define LOG(X) std::cout<<"[INFO] "<<(X)<<std::endl;
#define WARN(X) std::cout<<"[WARN] "<<(X)<<std::endl;
#define ERR(X) std::cerr<<"[ERROR] "<<(X)<<std::endl;

#else

#define LOG(X) 
#define WARN(X)
#define ERR(X)

#endif


namespace mlask {
class Model {
  private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::size_t epochs_;
    bool log_;

  public:
    Model() = default;
    Model(std::size_t size);
    Model(std::size_t epochs, bool log=true);
    Model(std::size_t size, std::size_t epochs, bool log=true);

    // adds a layer to a model
    void addLayer(std::unique_ptr<Layer> layer);

    // creates and adds fullyConnectedLayer with given in and out neurons
    template<std::size_t in, std::size_t out>
    void addFullyConnectedLayer();
    // creates and adds ActivationFunction layer with give function form and derived form
    void addActivationFunctionWithLambdas( actfunc func, actfunc derv);
    // creates and adds ActivationFunction layer of type activationFunction
    void addActivationFunction(InternalActivationFunction activationFunction);

    // performs backpropagation algorithm
    void backprop(vectorIn_ input, vectorOut_ expected, err_function err);
    // Fits the entire model(every layer added)
    void fit(float_t learning_rate);
    // calculates the output for given input
    vectorOut_ forward(vectorIn_ input) const;

    const Layer* getLayer(std::size_t index)const{ return layers_[index].get(); }
    const Layer* operator[](std::size_t index)const { return getLayer(index); }
};


template <std::size_t in, std::size_t out>
void Model::addFullyConnectedLayer(){
    layers_.push_back(std::make_unique<FullyConnectedLayer<in, out>>());
    LOG("Added FullyConnected Layer");
}

} // namespace mlask
