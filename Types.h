// Types.h
// type definitions

#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <functional>

// shorter types
typedef uint64_t U64;

// represents a piece on a grid
struct Piece {
    /*
        pinMask: pinMask of the piece
        bitboardIndex: which index is the piece in bitboardPointers
        colorBitboardIndex: which index is the color of the piece in bitboardPointers
        hasMoved: whether the piece has moved
    */

    U64 emptyU64 = 0;
    U64 pinMask = 0xffffffffffffffff;
    int bitboardIndex;
    int colorBitboardIndex;
    bool hasMoved = false;

    Piece(int _bitboardIndex, int _colorBitboardIndex) {
        bitboardIndex = _bitboardIndex;
        colorBitboardIndex = _colorBitboardIndex;
    }

    Piece() {
        hasMoved = true;
    }
};

// std::array with size equal to amount of objects
template<typename T, int N>
struct Array {
    /*
        size: number of set elements in the array
        array: the array
    */

    int size = 0;
    std::array<T, N> array;

    Array(int _size, std::array<T, N> _array) {
        size = _size;
        array = _array;
    }

    // get reference to array
    T& operator[](int index) {
        return array[index];
    }

    // "adds" an element to the array
    void push_back(T element) {
        array[size] = element;
        size++;
    }
};

// records types of moves
struct Moveset {
    /*
        normal: normal moves (all moves which are not specified below), first index is from, second index is to
        castle: castling moves, same structure as normal, but has a second set of moves
        enPassant: en passant moves, same structure as normal
        doubleMoves: double pawn moves, same structure as normal
        kingMoves: king moves, same structure as normal
    */

    Array<std::array<int, 2>, 218> normal = Array<std::array<int, 2>, 218>(0, {});
    Array<std::array<std::array<int, 2>, 2>, 2> castle = Array<std::array<std::array<int, 2>, 2>, 2>(0, {});
    Array<std::array<int, 2>, 2> enPassant = Array<std::array<int, 2>, 2>(0, {});
    Array<std::array<int, 3>, 8> doubleMoves = Array<std::array<int, 3>, 8>(0, {});


    Moveset() {

    }
};