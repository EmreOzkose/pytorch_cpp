#include <server.h>
#include <model.h>


int main(int argc, char** argv){
    std::cout << "start" << std::endl;

    auto server = SERVER();

    auto id2class = get_classes("../imagenet_classes.txt");
    auto alexnet = load_model("../alexnet.pt");
    
    auto img = server.receive_img(false);
    
    std::vector<torch::jit::IValue> inputs;
    
    auto img_tensor = Mat2Tensor(img);
    inputs.push_back(img_tensor);

    auto output = alexnet.forward(inputs).toTensor();
    auto prediction_index = output.argmax();

    std::cout << "Prediction: " << id2class[prediction_index.item<int>()] << std::endl;

    close(server.newsockfd);
    close(server.sockfd);

    return 0;
}
