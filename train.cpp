// train.cpp
// trains a NNUE model
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
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
    std::ifstream positions;
    std::ifstream evals;
    std::string fen;
    int eval;
    NNUE model = NNUE();
    model->to(device);

    torch::optim::Adam optimizer = torch::optim::Adam(model->parameters(), 0.001);
    torch::nn::MSELoss lossFunction = torch::nn::MSELoss();
    
    double runningLoss = 0;
    double lastLoss = 0;

    model->train();
    for (int epoch = 0; epoch < 2; epoch++) {
        int inputs = 0;
        evals.open("../data/evals.txt", std::ios_base::in);
        positions.open("../data/positions.txt", std::ios_base::in);
        while (std::getline(positions, fen)) {
            if (fen[0] == '\n') {
                continue;
            }
            if (inputs == 1000) {
                break;
            }
            evals >> eval;
            Position position;
            position.setFen(fen);
            std::array<torch::Tensor, 2> halfkp = position.halfkp();
            optimizer.zero_grad();

            torch::Tensor output = model->forward(halfkp[0], halfkp[1]).cuda();
            std::cout << output << "\n" << eval << std::endl; 
            std::vector<float> vec = {static_cast<float>(eval)};
            torch::Tensor loss = lossFunction(output, torch::from_blob(vec.data(), {1}, torch::TensorOptions().dtype(torch::kFloat)).cuda()).cuda();

            loss.backward();
            optimizer.step();

            runningLoss += loss.item().to<double>();
            inputs += 1;
            std::cout << inputs << std::endl;
        }
        std::cout << runningLoss / inputs << std::endl;
    }

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

    evals.close();
    positions.close();


    NNUE module;
    torch::load(module, "model.pt");
    output = module->forward(halfkp[0], halfkp[1]).cuda();
    std::cout << output << std::endl;

    return 0;
}