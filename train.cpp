// train.cpp
// trains a NNUE model
#include <ATen/core/TensorBody.h>
#include <ATen/ops/requires_grad_ops.h>
#include <ATen/ops/tensor_split.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/base.h>
#include <torch/data/datasets/tensor.h>
#include <torch/nn/modules/loss.h>
#include <torch/torch.h>
#include <iostream>
#include <ostream>
#include "Position.h"
#include "NNUE.h"
#include "CustomDataset.h"

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    std::string fen;
    int eval;
    std::string possibleEval;

    NNUE model = NNUE();
    model->to(device);

    torch::optim::Adam optimizer = torch::optim::Adam(model->parameters(), 0.0001);
    torch::nn::MSELoss lossFunction = torch::nn::MSELoss();
    
    double runningLoss = 0;
    double lastLoss = 0;
    std::vector<torch::Tensor> trainHalf1Data;
    std::vector<torch::Tensor> trainHalf2Data;
    std::vector<torch::Tensor> testHalf1Data;
    std::vector<torch::Tensor> testHalf2Data;
    std::vector<float> trainOutputData;
    std::vector<float> testOutputData;
    model->train();

    std::ifstream positions;
    std::ifstream evals;
    evals.open("../data/evals.txt", std::ios_base::in);
    positions.open("../data/positions.txt", std::ios_base::in);

    int inputs = 0;
    while (std::getline(positions, fen)) {
        Position position;
        position.setFen(fen);
        std::array<torch::Tensor, 2> halfkp = position.halfkp();

        if (inputs > 30000) {
            if (inputs == 33000) {
                break;
            }
            testHalf1Data.push_back(halfkp[0]);
            testHalf2Data.push_back(halfkp[1]);
            inputs += 1;
        } else {
            evals >> eval;

            
            float output = static_cast<float>(eval);
            trainHalf1Data.push_back(halfkp[0]);
            trainHalf2Data.push_back(halfkp[1]);
            trainOutputData.push_back(output);
            testOutputData.push_back(output);
            inputs += 1;
        }
    }

    auto trainDataset = CustomDataset(trainHalf1Data, trainOutputData, trainHalf2Data).map(Stack<Example3>());
    auto testDataset = CustomDataset(testHalf1Data, testOutputData, testHalf2Data).map(Stack<Example3>());
    auto trainDataloader = torch::data::make_data_loader(trainDataset, 64);
    auto testDataloader = torch::data::make_data_loader(testDataset, 64);

    for (int epoch = 0; epoch < 10; epoch++) {
        runningLoss = 0;
        inputs = 0;
        for (auto& batch : *trainDataloader) {
            optimizer.zero_grad();

            torch::Tensor outputs = torch::flatten(model(batch.data, batch.mask)).cuda();
            torch::Tensor loss = lossFunction(outputs, batch.target).cuda();
            loss.backward();
            
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1);
            optimizer.step();
            runningLoss += loss.item().to<double>();
            inputs += 1;            
        }
        std::cout << "Train loss: " << std::sqrt(runningLoss / inputs) << std::endl;
        runningLoss = 0;
        inputs = 0;
        for (auto& batch : *testDataloader) {
            torch::Tensor outputs = torch::flatten(model(batch.data, batch.mask)).cuda();
            torch::Tensor loss = lossFunction(outputs, batch.target).cuda();

            runningLoss += loss.item().to<double>();
            inputs += 1;            
        }

        std::cout << "Test loss: " << std::sqrt(runningLoss / inputs) << std::endl;
    }
    evals.close();
    positions.close();

    model->eval();
    Position position;
    position.setFen("3r2k1/5p2/p3p1p1/1P3q1p/2p3nP/5BP1/1P2QPK1/1R6 b - - 0 32");
    std::array<torch::Tensor, 2> halfkp = position.halfkp();
    torch::Tensor output = model(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;
    position.setFen("r3kbnr/pp3ppp/2n5/2pqN3/3Pp3/2P5/PP2bPPP/RNBQ1RK1 w kq - 0 9");
    halfkp = position.halfkp();
    output = model->forward(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;
    torch::save(model, "model.pt");

    // NNUE module;
    // torch::load(module, "model.pt");
    // output = module->forward(halfkp[0], halfkp[1]).cuda();
    // std::cout << output << std::endl;
    return 0;
}