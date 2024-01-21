// Bitbhoards.h
// Bitboard initializations

#pragma once

#include "Tables.h"
#include "Types.h"
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <stack>
#include <string>
#include <torch/torch.h>

class Position {
public:
    Position() {};
    void whitePawnMoves(Moveset& moveset);
    static U64 whitePawnAttacks(const U64& pawn);
    void blackPawnMoves(Moveset& moveset);
    static U64 blackPawnAttacks(const U64& pawn);
    void knightMoves(Moveset& moveset, const U64& own, U64 knights);
    static U64 knightAttacks(const int& index);
    void bishopMoves(Moveset& moveset, const U64& own, U64 bishops);
    static U64 bishopAttacks(const U64& blockMask, const int& index);
    static U64 xrayBishopAttacks(const U64& occ, const U64& blockers, const int& index);
    void rookMoves(Moveset& moveset, const U64& own, U64 rooks);
    static U64 rookAttacks(const U64& blockMask, const int& index);
    static U64 xrayRookAttacks(const U64& occ, const U64& blockers, const int& index);
    void queenMoves(Moveset& moveset, const U64& own, U64& queens);
    static U64 queenAttacks(U64& blockMask, int& index);
    static U64 kingAttacks(const int& index);
    bool whiteMoves(Moveset& moveset);
    bool blackMoves(Moveset& moveset);
    Position makeNormalMove(int& start, int& end);
    Position makeEnPassantMove(int& start, int& end, int& capture);
    Position makeCastlingMove(int& start1, int& end1, int& start2, int& end2);
    Position makeDoubleMove(int& start, int& end, int& canEnPassant);
    Position makePromotionMove(int& start, int& end, int& bitboardIndex);
    Position copy();
    U64 perft(int depth, std::stack<Position>& movelist);
    std::string toFen();
    void setFen(std::string fen);
    int orient(bool& turn, int& square);
    int halfkpIndex(bool& turn, int king, int& piece, int& index);
    std::array<torch::Tensor, 2> halfkp();
private:
    // intialize piece bitboards
    U64 whitePawns = 0x000000000000FF00;
    U64 blackPawns = 0x00FF000000000000;
    U64 whiteRooks = 0x0000000000000081;
    U64 blackRooks = 0x8100000000000000;
    U64 whiteKnights = 0x0000000000000042;
    U64 blackKnights = 0x4200000000000000;
    U64 whiteBishops = 0x0000000000000024;
    U64 blackBishops = 0x2400000000000000;
    U64 whiteKing = 0x0000000000000008;
    U64 blackKing = 0x0800000000000000;
    U64 whiteQueens = 0x0000000000000010;
    U64 blackQueens = 0x1000000000000000;

    // general bitboards
    U64 whitePieces = 0x000000000000FFFF;
    U64 blackPieces = 0xFFFF000000000000;
    U64 pieces = 0xFFFF00000000FFFF;
    U64 emptyBitboard = Tables::empty;
    U64 checkMask = Tables::full;

    // pointers to the bitboard for grid
    std::array<U64*, 16> bitboardPointers = {&whitePawns, &blackPawns, &whiteRooks, &blackRooks, &whiteKnights, &blackKnights, &whiteBishops, &blackBishops, &whiteKing, &blackKing, &whiteQueens, &blackQueens, &whitePieces, &blackPieces, &pieces, &emptyBitboard};

    // grid
    // each piece has the index of their bitboard pointers
    std::array<Piece, 64> grid = {Piece(2, 12), Piece(4, 12), Piece(6, 12), Piece(8, 12), Piece(10, 12), Piece(6, 12), Piece(4, 12), Piece(2, 12), 
                                  Piece(0, 12), Piece(0, 12), Piece(0, 12), Piece(0, 12), Piece(0, 12), Piece(0, 12), Piece(0, 12), Piece(0, 12), 
                                  Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), 
                                  Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), 
                                  Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), 
                                  Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), Piece(15, 15), 
                                  Piece(1, 13), Piece(1, 13), Piece(1, 13), Piece(1, 13), Piece(1, 13), Piece(1, 13), Piece(1, 13), Piece(1, 13), 
                                  Piece(3, 13), Piece(5, 13), Piece(7, 13), Piece(9, 13), Piece(11, 13), Piece(7, 13), Piece(5, 13), Piece(3, 13)};

    // special moves
    bool whiteQueenCastle = true;
    bool whiteKingCastle = true;
    bool blackQueenCastle = true;
    bool blackKingCastle = true;
    U64 enPassant = 0;

    // turn
    bool isWhiteTurn = true;

    int moveCount = 0;

    std::vector<U64> cache;
};

// helpers
void printBoard(U64 board);
void pushMoves(U64& targets, const int& index, Moveset& moveset);
int lsb(U64& board);
