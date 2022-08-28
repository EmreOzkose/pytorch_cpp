/**
 * @file main.cpp
 * @author Yunus Emre Ozkose (yunusemreozkose@gmail.com)
 * @brief 
 * 
 * Creating model
 * Creating Dataset
 * 
 */
#include <iostream>
#include <torch/torch.h>
#include "models.cpp"
#include "dataloader.cpp"

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

void xavier_init(torch::nn::Module& module) {
    // https://discuss.pytorch.org/t/libtorch-c-how-to-initialize-weights-xavier-in-a-sequential-module-with-apply-function/44920/3
    torch::NoGradGuard noGrad;
    if (auto* layer = module.as<torch::nn::Conv2d>()) {
        torch::nn::init::xavier_normal_(layer->weight);
        torch::nn::init::constant_(layer->bias, 0.01);
    }
}


int main(int argc, char** argv) {
    // set seed
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);

    // Define an MLP model
    // we will train a MLP in following blog, hence we can create a basic MLP now.
    MLP mlp;
    std::cout << mlp << std::endl;

    torch::Tensor example_input = torch::rand({1, 2});
    torch::Tensor output = mlp.forward(example_input);
    std::cout << output << std::endl;

    // Define a CNN model
    CNN cnn;
    cnn.apply(xavier_init);
    std::cout << cnn << std::endl;

    example_input = torch::rand({1, 3, 224, 224});
    output = cnn.forward(example_input);
    std::cout << output << std::endl;
    std::cout << output.sizes() << std::endl;

    // Create a Custom Dataset and get first sample.
    HouseDataset dataset;
    auto example = dataset.get(0);
    std::cout << example.data << std::endl;
    std::cout << example.target << std::endl;

    // Demo inference
    output = mlp.forward(example.data);
    std::cout << output << std::endl;

}