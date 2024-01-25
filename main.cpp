// Final Project
// Tony Tran
// chess engine
// main.cpp

#include <array>
#include <iostream>
#include <vector>
#include "Position.h"
#include "PVS.h"
#include "NNUE.h"

int main() {
    Position startPosition;
    
    startPosition.setFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    NNUE model;

    pvSearch(0, 0, 3, {}, model);
    return 0;
}