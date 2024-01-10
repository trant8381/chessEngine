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

    U64 nodes = 0;

    std::stack<Position> movelist;
    movelist.push(std::move(startPosition));

    std::cout << movelist.top().perft(6, movelist) << std::endl;

    return 0;
}