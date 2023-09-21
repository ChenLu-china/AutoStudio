#include <torch/torch.h>

#include "Dataset.h"
#include "../utils/Image.h"

#include <iostream>
using namespace std;


namespace AutoStudio
{

Dataset::Dataset(GlobalData *global_data):
    global_data_(global_data)
{
    // TO DO: add print info
    const auto& config = global_data_->config_["dataset"];
    const auto data_path = config["data_path"].as<std::string>();

    // Load images
    std::vector<Tensor> images;
    {
        // TO DO: add print info for image loading
        std::ifstream image_list(data_path + "/image_list.txt");
            for (int i = 0; i < n_images_; i++) {
            std::string image_path;
            std::getline(image_list, image_path);
            images.push_back(AutoStudio::Image::get_instance().ReadImageTensor(image_path).to(torch::kCPU));
        }
    }
}
Dataset::~Dataset(){

}

} // namespace AutoStudio