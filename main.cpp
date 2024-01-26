// Final Project
// Tony Tran
// chess engine
// main.cpp

#include <array>
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
    std::stack<Position> movelist;
    std::stack<Position> resMovelist;
    movelist.push(startPosition);
    
    std::cout << pvSearch(0, 0, 3, movelist, model, resMovelist) << std::endl;
    std::cout << resMovelist.top().toFen() << std::endl;
    return 0;
}