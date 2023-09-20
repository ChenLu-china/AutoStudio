#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "src/dataset/Dataset.h"
#include "src/utils/GlobalData.h"


int main(int argc, char* argv[]) {
    std::cout << "Go Go Auto Studio!!!" << std::endl;
    torch::manual_seed(2022);

    std::string conf_path = "./runtime_config.yaml";

    auto global_data_pool_ = std::make_unique<Auto_Studio::GlobalData>(conf_path);
    auto dataset = std::make_unique<Dataset>(global_data_pool_.get());
    return 0;
}