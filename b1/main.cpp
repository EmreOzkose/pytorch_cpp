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


torch::Tensor sigmoid_from_scratch(torch::Tensor input_tensor){
    return 1 / (1 + torch::exp(-input_tensor));
}

torch::Tensor binary_cross_entropy_from_scratch(torch::Tensor ground_truth, torch::Tensor prediction){
    torch::Tensor l = (1 - ground_truth) * torch::log(1 - prediction + 1e-8) + ground_truth * torch::log(prediction + 1e-8);
    return -1 * torch::mean(l);
}

// https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
torch::Tensor sigmoid_focal_loss(torch::Tensor input_tensor, torch::Tensor target_tensor, float alpha, float gamma){
    torch::Tensor p = sigmoid_from_scratch(input_tensor);
    torch::Tensor ce_loss = binary_cross_entropy_from_scratch(target_tensor, input_tensor);
    
    torch::Tensor p_t = p * target_tensor + (1 - p) * (1 - target_tensor);
    torch::Tensor loss = ce_loss * torch::pow((1 - p_t), gamma);

    if (alpha >= 0){
        torch::Tensor alpha_t = alpha * target_tensor + (1 - alpha) * (1 - target_tensor);
        loss = alpha_t * loss;
    }

    return loss.mean(0);
}

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

    // Example
    std::cout << "Sigmoid from scratch:\n" << sigmoid_from_scratch(tensor4) << "\n" << std::endl;

    torch::Tensor ground_truth = torch::tensor({1.0, 1.0, 0.0});
    torch::Tensor predictions1 = torch::tensor({1.0, 1.0, 0.0});
    torch::Tensor predictions2 = torch::tensor({1.0, 0.9, 0.0});
    torch::Tensor predictions3 = torch::tensor({1.0, 0.1, 0.0});

    std::cout << "Focal Loss pred 1: " << sigmoid_focal_loss(ground_truth, predictions1, 0.5, 2) << std::endl;
    std::cout << "Focal Loss pred 2: " << sigmoid_focal_loss(ground_truth, predictions2, 0.5, 2) << std::endl;
    std::cout << "Focal Loss pred 3: " << sigmoid_focal_loss(ground_truth, predictions3, 0.5, 2) << std::endl;

}