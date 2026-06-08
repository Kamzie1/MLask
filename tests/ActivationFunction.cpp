#include "LeakyRelu.hpp"
#include "Model.hpp"
#include "Relu.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"

using namespace mlask;

int main(){
    Model model(1, 4);
    model.addFullyConnectedLayer<1,4>();
    model.addLambdaActivationFunction([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
    model.addLayer<Relu>();
    model.addLayer<Sigmoid>();
    model.addLayer<LeakyRelu>(0.001);
    model.addLayer<Tanh>();
    std::cout<<model.str();
}
