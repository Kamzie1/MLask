#include "InternalActivationFunction.hpp"
#include "Relu.hpp"
#include "Model.hpp"

using namespace mlask;

int main(){
    Model model;
    model.addFullyConnectedLayer<1,1>();
    model.addActivationFunction(InternalActivationFunction::Relu);
    model.addActivationFunctionWithLambdas([](float_t x){ return x; }, [](float_t x){ return x; });
}
