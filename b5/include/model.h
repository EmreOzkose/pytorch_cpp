#include <iostream>
#include <fstream>
#include <map>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


torch::jit::script::Module load_model(char* model_path, bool doWarmUp = true){
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit(0);
    }

    if (doWarmUp){
        auto sample = torch::rand({1, 3, 224, 224});
        std::vector<torch::jit::IValue> inputs {sample};
        auto output = module.forward(inputs).toTensor();
    }

    return module;
}


std::map<int, std::string> get_classes(std::string path){
    std::map<int, std::string> id2class;
    

    std::ifstream ClassesFile(path);
    std::string line;

    int line_number = 0;
    while (getline (ClassesFile, line)) {
        id2class.insert(std::pair<int, std::string>(line_number, line));
        line_number++;
    }

    return id2class;
}


torch::Tensor normalize_tensor(torch::Tensor tensor_image){
    // normalize 3d tensor image
    // 
    auto mean1 = torch::tensor({0.485}).repeat({224, 224});
    auto mean2 = torch::tensor({0.456}).repeat({224, 224});
    auto mean3 = torch::tensor({0.406}).repeat({224, 224});
    auto mean = torch::stack({mean1, mean2, mean3});

    auto std1 = torch::tensor({0.229}).repeat({224, 224});
    auto std2 = torch::tensor({0.224}).repeat({224, 224});
    auto std3 = torch::tensor({0.225}).repeat({224, 224});
    auto std = torch::stack({std1, std2, std3});

    tensor_image = tensor_image / 255.0;
    tensor_image = (tensor_image - mean) / std;
    return tensor_image;
}


torch::Tensor Mat2Tensor(cv::Mat img){
    cv::resize(img, img, cv::Size(224, 224), 0, 0, 1);

    auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 });
    
    // Normalize Tensor
    tensor_image = normalize_tensor(tensor_image);

    tensor_image.unsqueeze_(0);

    return tensor_image.to(torch::kFloat);
}
