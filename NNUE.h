#pragma once

#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>

class NNUEImpl : public torch::nn::Module {
    public:
        NNUEImpl();

        torch::Tensor forward(torch::Tensor& half1, torch::Tensor& half2);
        std::vector<float> batchForward(std::vector<std::array<torch::Tensor, 2>>& split);
    private:
        torch::nn::Linear input1 = torch::nn::Linear(41024, 256);
        torch::nn::Linear input2 = torch::nn::Linear(41024, 256);
        torch::nn::ReLU clippedRelu1 = torch::nn::ReLU();
        torch::nn::ReLU clippedRelu2 = torch::nn::ReLU();

        torch::nn::Linear hiddenLayer1 = torch::nn::Linear(512, 32);
        torch::nn::Linear hiddenLayer2 = torch::nn::Linear(32, 32);
        torch::nn::ReLU activation1 = torch::nn::ReLU();
        torch::nn::ReLU activation2 = torch::nn::ReLU();

        torch::nn::Linear output = torch::nn::Linear(32, 1);
};

TORCH_MODULE(NNUE);