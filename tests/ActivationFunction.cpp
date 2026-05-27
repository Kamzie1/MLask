#include "Model.hpp"

using namespace mlask;

int main(){
    Model model(2, 4);
    model.addFullyConnectedLayer<2,4>();
    model.addActivationFunction(InternalActivationFunction::Relu);
    model.addActivationFunctionWithLambdas([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
    std::cout<<model.str();
}
