<a id="readme-top"></a>

<h3 align="center">MLask</h3>

  <p align="center">
    A custom c++ deep learning library.
    <br />
    <a href="https://kamzie1.github.io/MLask"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Kamzie1/MLask/blob/main/tests/OneNeuronNetwork.cpp">View Demo</a>
    &middot;
    <a href="https://github.com/Kamzie1/MLask/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
        <li>
            <a href="#defining-a-model">Defining a model</a>
            <ul>
                <li><a href="#adding-your-own-layer">Adding your own layer</a></li>
            </ul>
        </li>
        <li><a href="#training-the-model">Training the model</a></li>
        <li><a href="#exporting-the-model">Exporting the model</a></li>
        </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is a custom c++ deep learning library. It was meant to be a challenge to understand how all the professional ML libraries work under the hood. 

### Features
* Fully connected layers
* Activation functions
* Backpropagation algorithm
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
include(FetchContent)
FetchContent_Declare(MLask
    GIT_REPOSITORY https://gitlab.com/Kamzie1/MLask.git
    GIT_TAG 1.0.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(MLask)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Defining a model

You need to create a model with in, out, parameters. Optionally you can add number of layers, and whether to draw progress bar while learning.
Then you add layers.
```cpp
Model model(1, 1, 1,); // input of size 1, output of size 1, 1 layer
// You can add specific mlask layers
model.addFullyConnectedLayer<1, 1>(); // Correct output->input flow is validated
model.addActivationFunction(InternalActivationFunction::Relu); //Only builtin activation functions listed in InternalActivationFunctioon enum
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
model.addLayer(std::make_unique<CustomLayer>()); // or like this
```

### Training the model
```cpp
float_t learning_rate = 0.001;
for(std::size_t epochs=0; epochs < EPOCHS; epochs++){
  for(int x = 0;x<=SIZE;x++){
    model.backprop<DerivedStandardMean>(vectorIn{{(x-(SIZE/2.f))/(SIZE/2.f)}}, vectorOut{{Y[x]}}); // in arguments we construct an Eigen vector using mlask predefined type for simplicity
  }
  model.fit(learning_rate); // if your model has log set to true and it has correct number of epochs, then on fit it will update progress bar.
}
```

### Exporting the model
```cpp
    std::cout<<model.str();
    model.exportToONNX("onnx_format.onnx", "One Neuron Neural Network");
```

_For more examples, please refer to the [Documentation](https://kamzie1.github.io/MLask/)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
