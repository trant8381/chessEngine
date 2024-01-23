// train.cpp
// trains a NNUE model
#include <ATen/core/TensorBody.h>
#include <ATen/ops/requires_grad_ops.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/datasets/tensor.h>
#include <torch/nn/modules/loss.h>
#include <torch/torch.h>
#include <iostream>
#include <ostream>
#include "Position.h"
#include "NNUE.h"

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

    torch::optim::Adam optimizer = torch::optim::Adam(model->parameters(), 0.01);
    torch::nn::MSELoss lossFunction = torch::nn::MSELoss();
    
    double runningLoss = 0;
    double lastLoss = 0;
    std::vector<std::array<torch::Tensor, 2>> inputDataset;
    std::vector<float> outputDataset;
    model->train();

    std::ifstream positions;
    std::ifstream evals;
    evals.open("../data/evals.txt", std::ios_base::in);
    positions.open("../data/positions.txt", std::ios_base::in);

    int inputs = 0;
    while (std::getline(positions, fen)) {
        if (fen[0] == '\n') {
            continue;
        }
        if (inputs == 256) {
            break;
        }
        evals >> eval;

        Position position;
        position.setFen(fen);
        std::array<torch::Tensor, 2> halfkp = position.halfkp();
        float output = static_cast<float>(eval);
        std::array<torch::Tensor, 2> data = {std::move(halfkp[0]), std::move(halfkp[1])};
        inputDataset.push_back(data);
        outputDataset.push_back(output);

        inputs += 1;
    }

    int datasetSize = inputDataset.size();
    int batchSize = 64;
    std::vector<std::vector<torch::Tensor>> half1Split;
    std::vector<std::vector<torch::Tensor>> half2Split;
    std::vector<std::vector<float>> outputSplits;

    for (int i = 0; i < datasetSize; i++) {
        if (i % batchSize == 0) {
            half1Split.push_back({});
            half2Split.push_back({});
            outputSplits.push_back({});
        }
        half1Split[i / batchSize].push_back(inputDataset[i][0]);
        half2Split[i / batchSize].push_back(inputDataset[i][1]);
        outputSplits[i / batchSize].push_back(outputDataset[i]);
    }

    for (int epoch = 0; epoch < 20; epoch++) {
        runningLoss = 0;
        for (int i = 0; i < outputSplits.size(); i++) {
            std::vector<torch::Tensor>& half1 = half1Split[i];
            std::vector<torch::Tensor>& half2 = half2Split[i];
            torch::Tensor half1batch = torch::stack(half1);
            torch::Tensor half2batch = torch::stack(half2);
            optimizer.zero_grad();
            std::cout << half1batch.size(0) << std::endl;
            std::cout << half1batch.size(1) << std::endl;
            std::cout << half2batch.size(0) << std::endl;
            std::cout << half2batch.size(1) << std::endl;
            torch::Tensor outputs = model(half1batch, half2batch).cuda();
            // std::cout << output << "\n" << eval << std::endl; 
            std::vector<float> evals = outputSplits[i];
            std::cout << outputs << std::endl;
            // std::cout << output << std::endl;
            std::cout << torch::from_blob(evals.data(), {static_cast<long>(evals.size())}, torch::TensorOptions().dtype(torch::kFloat)) << std::endl;

            torch::Tensor loss = lossFunction(outputs,
             torch::from_blob(evals.data(), {static_cast<long>(evals.size())}, torch::TensorOptions().dtype(torch::kFloat)).cuda()).cuda();

            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1);
            optimizer.step();

            runningLoss += loss.item().to<double>();
            inputs += 1;
            std::cout << runningLoss / inputs << std::endl;
            std::cout << epoch << std::endl;
            
        }
    }
    evals.close();
    positions.close();

    model->eval();
    Position position;
    position.setFen("3r2k1/5p2/p3p1p1/1P3q1p/2p3nP/5BP1/1P2QPK1/1R6 b - - 0 32");
    std::array<torch::Tensor, 2> halfkp = position.halfkp();
    torch::Tensor output = model->forward(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;
    position.setFen("r3kbnr/pp3ppp/2n5/2pqN3/3Pp3/2P5/PP2bPPP/RNBQ1RK1 w kq - 0 9");
    halfkp = position.halfkp();
    output = model->forward(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;
    torch::save(model, "model.pt");

    NNUE module;
    torch::load(module, "model.pt");
    output = module->forward(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;
    return 0;
}