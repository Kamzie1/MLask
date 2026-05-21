#include "Model.hpp"

using namespace mlask;

int main(){
    Model model;
    model.addFullyConnectedLayer<1,1>();
    model.addActivationFunction(InternalActivationFunction::Relu);
}
