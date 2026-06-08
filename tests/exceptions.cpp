#include "Model.hpp"
#include "Relu.hpp"

using namespace mlask;

class CustomLayer : public Layer{
public:
    // You need to override these functions. However if you want your layer to be exportable to ONNX for example, you also need to override tryConvertToONNX.
    vectorOut forward(vectorIn input) override{return input; }
    vectorIn backward(vectorOut error) override{ return error; }
    void fit(float_t learning_rate) override {}
};



int main(){
    Model model(1, 4);
    model.addFullyConnectedLayer<1,4>();
    model.addLambdaActivationFunction([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } );
    model.addLayer<Relu>();
    model.addLayer<CustomLayer>();// You can add your custom layer to the model like this.
    model.forward(vectorIn{{0}});
    try{
        model.exportToONNX("error");
    }catch(ExportError& e){
    }
    catch(ArchitectureError& e){
    }
}
