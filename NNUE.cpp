#include "NNUE.h"
#include <ATen/core/TensorBody.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

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
    torch::Tensor transformed = torch::concat({input->forward(clippedRelu->forward(half1)),
                                               input->forward(clippedRelu->forward(half2))});
    
    return output->forward(activation2->forward(hiddenLayer2->forward(activation1->forward(hiddenLayer1->forward(transformed)))));
}