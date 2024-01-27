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
    std::string fen;
    torch::load(model, "model.pt");
    model->to(torch::Device(torch::kCUDA));
    std::cout << "Enter fen: ";
    std::cin >> fen;
    startPosition.setFen(fen);
    bool opponentTurn;
    std::cout << "Is it opponent's turn to play?";
    std::cin >> opponentTurn;
    std::stack<Position> movelist;
    movelist.push(startPosition);
    while (true) {
        Position position = movelist.top();
        Moveset moveset;
        if (position.isWhiteTurn) {
            position.whiteMoves(moveset);
        } else {
            position.blackMoves(moveset);
        }

        if ((moveset.castle.size == 0 && moveset.doubleMoves.size == 0 && moveset.enPassant.size == 0 && moveset.normal.size == 0 && moveset.promotion.size == 0)) {
            
        }

        if (opponentTurn) {
            std::string move;
            std::stringstream ssMove;
            std::cout << "Please make a move." << std::endl;
            std::getline(std::cin, move);
            ssMove << move;
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
                case 'c':
                    int start2;
                    int end2;
                    ssMove >> start >> end >> start2 >> end2;
                    movelist.push(position.makeCastlingMove(start, end, start2, end2));
                    break;
                case 'p':
                    int index;
                    ssMove >> start >> end >> index;
                    movelist.push(position.makePromotionMove(start, end, index));
                    break;
            }
        } else {
            std::stack<Position> resMovelist = movelist;
            std::cout << pvSearch(-1000000, 1000000, 3, movelist, model, resMovelist) << std::endl;
            std::cout << resMovelist.top().toFen() << std::endl;
        }

        opponentTurn = !opponentTurn;
    }
    return 0;
}