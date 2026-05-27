#include "Model.hpp"

using namespace mlask;

int main(){
    Model model(1, 1);
    model.addFullyConnectedLayer<1,1>();
    model.addActivationFunction(InternalActivationFunction::Relu);
    model.addActivationFunctionWithLambdas([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
}
