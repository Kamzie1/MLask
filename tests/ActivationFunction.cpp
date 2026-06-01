#include "Model.hpp"
#include "Relu.hpp"

using namespace mlask;

int main(){
    Model model(2, 4);
    model.addFullyConnectedLayer<2,4>();
    model.addActivationFunction(InternalActivationFunction::Relu);
    model.addLambdaActivationFunction([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
    model.addLayer<Relu>();
    std::cout<<model.str();
}
