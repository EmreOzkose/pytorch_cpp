/**
 * @file main.cpp
 * @author Yunus Emre Ozkose (yunusemreozkose@gmail.com)
 * @brief 
 * 
 * House Price Prediction
 * 
 */
#include <ctime>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>


struct MLP : torch::nn::Module {
    // This class is not used, but written for only showing how to define our House Price Model in CPP. 
    // We will load Jit model, hence it is not necessary to define a model in CPP, like below (MLP()). 
    // It is because we save Pytorch Model by tracing it. That means, we create a model graph which is called traced model.

    MLP():
        fc1(2, 10),
        fc2(10, 1) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

    torch::Tensor forward(torch::Tensor x) {
        x = fc1->forward(x);
        x = torch::relu(x);

        x = fc2->forward(x);
        x = torch::relu(x);

        return x;
    }

  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};


torch::jit::script::Module load_model(std::string model_path){
    // https://pytorch.org/tutorials/advanced/cpp_export.html#a-minimal-c-application
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        std::cout << "model loaded from " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
    return module;
}


int main(int argc, char** argv) {
    // set seed
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);

    std::string model_path = "../traced_house_price_prediction.pt";
    torch::jit::script::Module model = load_model(model_path);

    std::cout << "inference for a house where lot area is " << argv[1] << " and built year is " << argv[2] << std::endl;

    // Create input tensor
    // taken from python script, statistics dictionary
    float lot_area_mean = 10597.72089041096;
    float lot_area_std = 10684.958322516175;
    float built_year_mean = 1971.1207191780823;
    float built_year_std = 30.27955974415448;

    float lot_area_in = (atoi(argv[1]) - lot_area_mean) / lot_area_std;
    float built_year_in = (atoi(argv[2]) - built_year_mean) / built_year_std;
    torch::Tensor input = torch::tensor({lot_area_in , built_year_in});

    // Inference
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Execute the model and turn its output into a tensor.
    torch::Tensor output = model.forward(inputs).toTensor();
    std::cout << "Prediction: " << output.item() << '\n';
}