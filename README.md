<a id="readme-top"></a>

<h3 align="center">MLask</h3>

  <p align="center">
    A custom c++ deep learning library.
    <br />
    <a href="https://kamzie1.github.io/MLask"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="https://github.com/Kamzie1/MLask-Demo">View Demo</a>
    &middot;
    <a href="https://github.com/Kamzie1/MLask/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

This is a custom c++ deep learning library. It was meant to be a challenge to understand how all the professional ML libraries work under the hood. 

### Features
* Fully connected layers
* Activation functions
* Backpropagation algorithm
* Confusion Matrixs
* Export to ONNX format

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [CMake](https://cmake.org)
* [Eigen](https://gitlab.com/libeigen/eigen)
* [ONNX](https://github.com/onnx/onnx)
* [Protobuf](https://github.com/protocolbuffers/protobuf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* CMake 3.28 or higher
* c++23 compatible compiler

### Installation

```cmake
# add these to install faster, however protobuf is pretty big, so it will take a while. Be patient.
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(MLask
    GIT_REPOSITORY https://gitlab.com/Kamzie1/MLask.git
    GIT_TAG 1.0.3
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(MLask)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Defining a model

You need to create a model. Then you add layers.
```cpp
Model model(1, 1);

model.addFullyConnectedLayer<1, 1>(); // Correct output->input flow is validated
model.addLayer<Relu>();
model.addLambdaActivationFunction([](float_t x){ return 0.5*x*x; }, [](float_t x){ return x; } ); // A layer that should be used only for prototyping.
```

#### Adding your own layer

You can define and add your own layer. All you need to do is create a class that derives from abstract Layer class.

```cpp
class CustomLayer : public Layer{
public:
    // You need to override these functions. However if you want your layer to be exportable to ONNX for example, you also need to override tryConvertToONNX.
    vectorOut forward(vectorIn input) override;
    vectorIn backward(vectorOut error) override;
    void fit(float_t learning_rate) override {}
};

model.addLayer<CustomLayer>();// You can add your custom layer to the model like this.
```

To create activation function you can derive from ActivationFunction, which makes the implementation easier.

### Training the model
```cpp
  model.fit<errFunction>(trainX, trainY, 1000, 0.001);
  ConfusionMatrix conf = model.evaluate(testX, testY, interpret) // interpret is a ptr to the function that interprets the solution.
  std::cout<<conf;
```

### Exporting the model
```cpp
    std::cout<<model;
    model.exportToONNX("onnx_format.onnx", "One Neuron Neural Network");
```

_For more examples, please refer to the [Documentation](https://kamzie1.github.io/MLask/)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>
