#include "NNUE.h"
#include <ATen/core/TensorBody.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>
#include <vector>

NNUEImpl::NNUEImpl() {
    register_module("input1", input1);
    register_module("input2", input2);
    register_module("clippedRelu1", clippedRelu1);
    register_module("clippedRelu2", clippedRelu2);
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

    torch::Tensor transformed = torch::concat({clippedRelu1(input1(half1)),
                                               clippedRelu2(input2(half2))});
    std::cout << "here" << std::endl;
    return output(activation2(hiddenLayer2(activation1(hiddenLayer1(transformed)))));
}

std::vector<float> NNUEImpl::batchForward(std::vector<std::array<torch::Tensor, 2>>& split) {
    std::vector<float> outputs;
    for (std::array<torch::Tensor, 2>& halfkp : split) {
        torch::Tensor transformed = torch::concat({input1->forward(clippedRelu1->forward(halfkp[0])),
                                                   input2->forward(clippedRelu2->forward(halfkp[1]))});
        
        outputs.push_back(output->forward(
                            activation2->forward(
                                hiddenLayer2->forward(
                                    activation1->forward(
                                        hiddenLayer1->forward(transformed)))))[0].item().to<float>());
    }

    return outputs;
}

