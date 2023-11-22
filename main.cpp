#include <iostream>
#include <memory>
#include <torch/torch.h>
// #include "src/dataset/Dataset.h"
// #include "src/utils/GlobalData.h"
#include "src/pipeline/Pipeline.h"

int main(int argc, char* argv[]) {
    std::cout << "Go Go Auto Studio!!!" << std::endl;
    torch::manual_seed(2022);

    std::string conf_path = "./runtime_config.yaml";

    // auto global_data_pool_ = std::make_unique<AutoStudio::GlobalData>(conf_path);
    // auto dataset = std::make_unique<AutoStudio::Dataset>(global_data_pool_.get());
    auto runner = std::make_unique<AutoStudio::Runner>(conf_path);
    runner->Execute();
    return 0;
}