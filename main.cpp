// Final Project
// Tony Tran
// chess engine
// main.cpp

#include <array>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "Position.h"
#include "PVS.h"
#include "NNUE.h"

int main() {
    Position startPosition;
    
    startPosition.setFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    NNUE model;
    torch::load(model, "model20k.pt");
    model->to(torch::Device(torch::kCUDA));
    std::stack<Position> movelist;
    movelist.push(startPosition);
    std::stack<Position> resMovelist = movelist;
    std::cout << pvSearch(-1000000, 1000000, 10, movelist, model, resMovelist) << std::endl;
    std::cout << resMovelist.top().toFen() << std::endl;
    return 0;
}