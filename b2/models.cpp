#include <torch/torch.h>

struct MLP : torch::nn::Module {
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


struct CNN : torch::nn::Module {
    CNN():
        conv1(torch::nn::Conv2dOptions(3, 16, /*kernel_size=*/5)),
        pooling1(torch::nn::AvgPool2dOptions({2,2})),
        conv2(torch::nn::Conv2dOptions(16, 32, /*kernel_size=*/3)),
        fc(373248, 1) {
            register_module("conv1", conv1);
            register_module("pooling1", pooling1);
            register_module("conv2", conv2);
            register_module("fc", fc);
        }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = torch::relu(x);
        x = pooling1->forward(x);
        
        x = conv2->forward(x);
        x = torch::relu(x);

        x = x.flatten();
        x = fc->forward(x);
        x = torch::relu(x);

        return x;
    }

    torch::nn::Conv2d conv1;
    torch::nn::AvgPool2d pooling1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc;
};

