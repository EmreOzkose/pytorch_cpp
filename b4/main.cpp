#include <iostream>
#include <fstream>
#include <map>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

torch::jit::script::Module load_model(char* model_path){
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit(0);
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


torch::Tensor read_image(std::string image_path){
    cv::Mat img = cv::imread(image_path);
    cv::resize(img, img, cv::Size(224, 224), 0, 0, 1);

    auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 });
    
    // Normalize Tensor
    tensor_image = normalize_tensor(tensor_image);

    tensor_image.unsqueeze_(0);

    return tensor_image.to(torch::kFloat);
}

int main(int argc, char** argv) {
    // set seed
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);

    auto id2class = get_classes("../imagenet_classes.txt");
    auto alexnet = load_model("../alexnet.pt");

    std::string image_path = argv[1];
    auto image = read_image(image_path);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);

    auto output = alexnet.forward(inputs).toTensor();
    auto prediction_index = output.argmax();
    
    std::cout << "Prediction: " << id2class[prediction_index.item<int>()] << std::endl;

}
