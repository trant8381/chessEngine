// Final Project
// Tony Tran
// chess engine
// main.cpp

#include "Position.h"
#include <array>
#include <iostream>
#include <vector>

int main() {
    Position startPosition;
    
    startPosition.setFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    U64 nodes = 0;

    std::stack<Position> movelist;
    movelist.push(std::move(startPosition));

    std::cout << movelist.top().perft(3, movelist) << std::endl;

    return 0;
}