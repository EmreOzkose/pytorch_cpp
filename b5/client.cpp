#include <client.h>


int main(int argc, char** argv){
	auto client = CLIENT();

	std::string image_path = "/home/sestek/workspace/medium/cpp_series/data/bee_ant/hymenoptera_data/bees/16838648_415acd9e3f.jpg";
    cv::Mat image = cv::imread(image_path);
	cv::resize(image, image, cv::Size( IM_WIDTH , IM_HEIGHT ));

	client.send_image(image);

	close(client.sockfd);

	return 0;
}
