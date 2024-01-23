#include "NNUE.h"
#include <ATen/core/TensorBody.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>
#include <vector>

NNUEImpl::NNUEImpl() {
    register_module("input", input);
    register_module("clippedRelu", clippedRelu);
    register_module("hiddenLayer1", hiddenLayer1);
    register_module("hiddenLayer2", hiddenLayer2);
    register_module("activation1", activation1);
    register_module("activation2", activation2);
    register_module("output", output);
}

torch::Tensor NNUEImpl::forward(torch::Tensor& half1, torch::Tensor& half2) {
    std::cout << half1.size(0) << std::endl;
    std::cout << half1.size(1) << std::endl;
    std::cout << half2.size(0) << std::endl;
    std::cout << half2.size(1) << std::endl;
    torch::Tensor transformed = torch::concat({input->forward(clippedRelu->forward(half1)),
                                               input->forward(clippedRelu->forward(half2))});
    
    return output->forward(activation2->forward(hiddenLayer2->forward(activation1->forward(hiddenLayer1->forward(transformed)))));
}

std::vector<float> NNUEImpl::batchForward(std::vector<std::array<torch::Tensor, 2>>& split) {
    std::vector<float> outputs;
    for (std::array<torch::Tensor, 2>& halfkp : split) {
        torch::Tensor transformed = torch::concat({input->forward(clippedRelu->forward(halfkp[0])),
                                                input->forward(clippedRelu->forward(halfkp[1]))});
        
        outputs.push_back(output->forward(
                            activation2->forward(
                                hiddenLayer2->forward(
                                    activation1->forward(
                                        hiddenLayer1->forward(transformed)))))[0].item().to<float>());
    }

    return outputs;
}

