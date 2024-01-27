// Final Project
// Tony Tran
// chess engine
// main.cpp

#include <array>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <torch/torch.h>
#include "Position.h"
#include "PVS.h"
#include "NNUE.h"

int main() {
    Position startPosition;
    NNUE model;
    torch::load(model, "model20k.pt");
    model->to(torch::Device(torch::kCUDA));
    startPosition.setFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    bool opponentTurn;
    std::cout << "Is it opponent's turn to play?";
    std::cin >> opponentTurn;
    std::stack<Position> movelist;
    movelist.push(startPosition);

    if (opponentTurn) {
        Position position = movelist.top();
        std::string move;
        std::stringstream ssMove;
        ssMove << move;
        std::cout << "Please make a move." << std::endl;
        std::getline(std::cin, move);
        int start;
        int end;

        switch (move[0]) {
            case 'n':
                ssMove >> start >> end;
                movelist.push(position.makeNormalMove(start, end));
                break;
            case 'e':
                int capture;
                ssMove >> start >> end >> capture;
                movelist.push(position.makeEnPassantMove(start, end, capture));
                break;
            case 'd':
                int enPassant;
                ssMove >> start >> end;
                movelist.push(position.makeDoubleMove(start, end, enPassant));
                break;

        }
    } else {
        std::stack<Position> resMovelist = movelist;
        std::cout << pvSearch(-1000000, 1000000, 3, movelist, model, resMovelist) << std::endl;
        std::cout << resMovelist.top().toFen() << std::endl;
    }
    return 0;
}