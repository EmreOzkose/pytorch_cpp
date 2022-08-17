/**
 * @file main.cpp
 * @author Yunus Emre Ozkose (yunusemreozkose@gmail.com)
 * @brief 
 * 
 * Basic Tensor operations with Aten and Torch
 * Full functionality can be seen in https://pytorch.org/cppdocs/api/namespace_at.html#functions
 * 
 */
#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>

int main() {
    // set seed
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);

    // Define some random Tensors
    torch::Tensor tensor1 = torch::rand({2, 3});
    torch::Tensor tensor2 = torch::rand({2, 3});
    std::cout << "tensor1:\n" << tensor1 << std::endl;
    std::cout << "tensor2:\n" << tensor2 << std::endl;
    std::cout << std::endl;

    // Basic operations
    torch::Tensor addition = tensor1 + tensor2;
    torch::Tensor multiplication = tensor1 * tensor2;
    torch::Tensor matrix_multiplication = tensor1.matmul(tensor2.transpose(1, 0));
    
    std::cout << "addition:\n" << addition << "\n" << std::endl;
    std::cout << "multiplication:\n" << multiplication << "\n" << std::endl;
    std::cout << "matrix_multiplication:\n" << matrix_multiplication << "\n" << std::endl;
    
    // Squeeze / Unsqueeze
    std::cout << "original shape:\n" << tensor1.sizes() << "\n" << std::endl;
    std::cout << "unsqueeze shape:\n" << tensor1.unsqueeze(0).sizes() << "\n" << std::endl;
    std::cout << "3 unsqueeze + 1 squeeze shape:\n" << tensor1.unsqueeze(0).unsqueeze(0).unsqueeze(0).squeeze(0).sizes() << "\n" << std::endl;

    // Flatten
    std::cout << "flattened:\n" << tensor1.flatten() << "\n" << std::endl;
    std::cout << "flatten shape:\n" << tensor1.flatten().sizes() << "\n" << std::endl;

    // math
    torch::Tensor tensor3 = torch::range(0, 5, 1);
    std::cout << "Range:\n" << tensor3 << "\n" << std::endl;

    std::cout << "Square:\n" << tensor3.square() << "\n" << std::endl;
    std::cout << "Square root:\n" << tensor3.sqrt() << "\n" << std::endl;
    
    std::cout << "sum:\n" << tensor3.sum() << "\n" << std::endl;
    std::cout << "mean:\n" << tensor3.mean() << "\n" << std::endl;
    std::cout << "Standart Deviation:\n" << tensor3.std() << "\n" << std::endl;
    
    // Stacking
    std::cout << "tensor shapes:\n" << tensor1.sizes() << "," << tensor2.sizes() << "\n" << std::endl;
    std::cout << "vertical stack:\n" << torch::vstack({tensor1, tensor2}).sizes() << "\n" << std::endl;
    std::cout << "horizontal stack:\n" << torch::hstack({tensor1, tensor2}).sizes() << "\n" << std::endl;
    
    // Device Ops
    std::cout << "Tensor Device: " << tensor1.get_device() << std::endl;
    std::cout << "Is CUDA available?: " << torch::cuda::is_available() << std::endl;
    std::cout << "Number of available devices: " << torch::cuda::device_count() << std::endl;
    if (torch::cuda::is_available() > 0){
        for (int i=0; i<torch::cuda::device_count(); i++){
            std::cout << "Properties of device " << i << ":\n" << torch::cuda::getDeviceProperties(i) << std::endl;
        }
    }
    //std::cout << "Tensor Device:\n" << tensor1.cpu().get_device() << "\n" << std::endl;
    //std::cout << "Tensor Device:\n" << tensor1.cuda().get_device() << "\n" << std::endl;

    // Non-linearity
    torch::Tensor tensor4 = torch::range(-1, 5, 1);

    std::cout << "Log Softmax:\n" << tensor4.log_softmax(0) << "\n" << std::endl;
    std::cout << "ReLU:\n" << tensor4.relu() << "\n" << std::endl;
    std::cout << "Sigmoid:\n" << tensor4.sigmoid() << "\n" << std::endl;
    std::cout << "Tanh:\n" << tensor4.tanh() << "\n" << std::endl;

}