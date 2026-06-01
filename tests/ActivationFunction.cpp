#include "Model.hpp"
#include "Relu.hpp"

using namespace mlask;

int main(){
    Model model(1, 4);
    model.addFullyConnectedLayer<1,4>();
    model.addActivationFunction(InternalActivationFunction::Relu);
    model.addLambdaActivationFunction([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
    model.addLayer<Relu>();
    model.forward(vectorIn{{0}});
}
