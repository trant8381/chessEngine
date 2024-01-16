// tableGenerator.cpp
// script to generate tables in Tables.h
// not intended to be efficient

#include "Position.h"
#include "Tables.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>
#include <array>
#include <random>

// debugging function to print bitboards to file
void printBoard(U64 board, std::ofstream& file) {
    /*
        board: bitboard to display
        file: output file
    */

    std::string binary = "";
    U64 temp = board;

    // number to binary
    while (temp != 0) {
        int mod = temp % 2;
        binary += std::to_string(mod);
        temp = temp / 2;
    }

    // add some zeroes
    int end = 64 - binary.length();
    for (int i = 0; i < end; i++) {
        binary += "0";
    }

    std::reverse(binary.begin(), binary.end());

    // print board
    for (int i = 0; i < 64; i++) {
        file << binary[i];
        if ((i + 1) % 8 == 0) {
            file << std::endl;
        }
    }

    file << std::endl;
}

// generate knight moves
void generateKnightBitboards(std::ofstream& file) {
    // file: output file

    file << "{";

    // loop through all bitboards with one set bit
    for (U64 x = 0; x < 64; x++) {
        U64 i = std::pow(2, x);
        // all knight moves using shifts
        U64 a = ((i & Tables::notFileH) << 15);
        U64 b = ((i & Tables::notFileA) << 17);
        U64 c = ((i & Tables::notFileAB) << 10);
        U64 d = ((i & Tables::notFileGH) << 6);
        U64 e = ((i & Tables::notFileAB) >> 6);
        U64 f = ((i & Tables::notFileGH) >> 10);
        U64 g = ((i & Tables::notFileH) >> 17);
        U64 h = ((i & Tables::notFileA) >> 15);

        // debugging
        // printBoard(i);
        // printBoard(a | b | c | d | e | f | g | h);

        file << "0x";
        file << std::hex << (a | b | c | d | e | f | g | h); // combine all shifts to get moves
        file << ", ";
    }
    file << "}";
}

// generate king moves
void generateKingBitboards(std::ofstream& file) {
    // file: output file

    file << "{";

    // loop through all bitboards with one true bit
    for (U64 x = 0; x < 64; x++) {
        U64 i = std::pow(2, x);
        // get king moves using shifts
        U64 a = ((i & Tables::notFileA) << 1);
        U64 b = ((i & Tables::notFileH) >> 1);
        U64 c = ((i) << 8);
        U64 d = ((i) >> 8);
        U64 e = ((i & Tables::notFileA) >> 7);
        U64 f = ((i & Tables::notFileH) >> 9);
        U64 g = ((i & Tables::notFileH) << 7);
        U64 h = ((i & Tables::notFileA) << 9);

        // debugging
        // printBoard(i);
        // printBoard(a | b | c | d | e | f | g | h);

        file << "0x";
        file << std::hex << (a | b | c | d | e | f | g | h); // combine all shifts to get moves
        file << ", ";
    }
    file << "}";
}

// generates magic numbers
U64 generateMagicNumbers(int compound, std::vector<U64>& permutations, std::vector<U64>& attacks, std::vector<std::vector<U64>>& results, int size) {
    /*
        compound: amount of times to repeat masking random numbers for a magic number
        permutations: the permutations of blockers
        attacks: the corresponding attacks of the permutations
        results: output table of rook attacks
        size: how many possible attacks there are not including blockers

        output: a magic number
    */

    // random number generator for 64 bit integers
    std::random_device device;
    std::mt19937_64 generator(device());
    std::uniform_int_distribution<U64> distribution;

    std::vector<U64> lookupTable(4096, 0);
    U64 magicNum = Tables::full;
    
    // generate magic numbers
    while (true) {
        // random number with low amount of set bits
        for (int _ = 0; _ < compound; _++) {
            magicNum = magicNum & distribution(generator);
        }
        // clear lookup table
        std::fill(lookupTable.begin(), lookupTable.end(), 0);
        bool failure = false;

        for (int d = 0; d < permutations.size(); d++) {
            // generate index
            U64 index = (permutations[d] * magicNum) >> (64 - Tables::rookMagicNumberLengths[size]);

            // check for bad collisions
            if (lookupTable[index] == 0) {
                lookupTable[index] = attacks[d];
            } else if (lookupTable[index] != attacks[d]) {
                // set failure flag
                failure = true;
                break;
            }
        }

        if (!failure) {
            results.push_back(lookupTable);
            break;
        }
    }

    return magicNum;
}

// generate pieces able to block rook
void generateRookBlockerBitboards(std::ofstream& file) {
    // file: output file

    file << "{";

    U64 multiplier = 1 + std::pow(2, 8) + std::pow(2, 16) + std::pow(2, 24) + std::pow(2, 32)
        + std::pow(2, 40) + std::pow(2, 48) + std::pow(2, 56); // the multiplier to get cols
    U64 oneRow = 0x00000000000000FF; // a basic row

    for (U64 i = 0; i < 64; i++) {
        U64 piece = std::pow(2, i);
        U64 row = (oneRow << (i / 8) * 8); // shift basic row by applicable amount

        U64 firstCol = (std::pow(2, i % 8)); // the first bit of the column
        U64 topCol = (std::pow(2, (i % 8))) * multiplier; // the rest of the column
        U64 col = (topCol) | firstCol;

        // clear first and last, as they are always blockers
        U64 clearIndexRow1 = __builtin_ctzll(row);
        U64 clearIndexRow2 = 63 - __builtin_clzll(row);
        U64 clearIndexCol1 = __builtin_ctzll(col);
        U64 clearIndexCol2 = 63 - __builtin_clzll(col);

        // clear bits
        col = col & (~(1ULL << clearIndexCol1));
        col = col & (~(1ULL << clearIndexCol2));
        row = row & (~(1ULL << clearIndexRow1));
        row = row & (~(1ULL << clearIndexRow2));

        file << "0x";
        file << std::hex << ((row | col) & (~piece)); // disclude piece location
        file << ", ";
    }
    file << "}";
}

// generate rook magic numbers and rook attack tables
void generateRookMagics(std::ofstream& file) {
    // file: output file

    file << "{";

    U64 multiplier = 1 + std::pow(2, 8) + std::pow(2, 16) + std::pow(2, 24) + std::pow(2, 32)
        + std::pow(2, 40) + std::pow(2, 48) + std::pow(2, 56); // the multiplier to get cols
    U64 oneRow = 0x00000000000000FF; // a basic row

    std::vector<std::vector<U64>> rookAttacks;

    for (U64 i = 0; i < 64; i++) {
        U64 piece = 1ULL << i;
        U64 row = (oneRow << (i / 8) * 8); // shift basic row by applicable amount

        U64 firstCol = (std::pow(2, i % 8)); // the first bit of the column
        U64 topCol = (std::pow(2, (i % 8))) * multiplier; // the rest of the column
        U64 col = (topCol) | firstCol;

        // clear first and last, as they are always blockers
        U64 clearIndexRow1 = __builtin_ctzll(row);
        U64 clearIndexRow2 = 63 - __builtin_clzll(row);
        U64 clearIndexCol1 = __builtin_ctzll(col);
        U64 clearIndexCol2 = 63 - __builtin_clzll(col);

        // clear bits
        col = col & (~(1ULL << clearIndexCol1)) & (~piece);
        col = col & (~(1ULL << clearIndexCol2));
        row = row & (~(1ULL << clearIndexRow1) & (~piece));
        row = row & (~(1ULL << clearIndexRow2));
        
        
        std::vector<U64> posCol = {};
        std::vector<U64> posRow = {};
        U64 copy = col;

        // get the positions of the columns
        while (copy != 0) {
            int index = __builtin_ctzll(copy);
            posCol.push_back(std::pow(2, index));
            copy = copy & (~(1ULL << index));
        }

        copy = row;

        // get the positions of the rows
        while (copy != 0) {
            int index = __builtin_ctzll(copy);
            posRow.push_back(std::pow(2, index));
            copy = copy & (~(1ULL << index));
        }

        std::vector<U64> colPerms;
        std::vector<U64> rowPerms;
        std::vector<U64> colAttacks;
        std::vector<U64> rowAttacks;
        U64 colPermIndex = 0;
        U64 rowPermIndex = 0;
        U64 colSize = posCol.size();
        U64 rowSize = posRow.size();

        // get all the permutations of the columns
        while (colPermIndex != std::pow(2, colSize)) {
            // this index shows which column bits are set
            copy = colPermIndex;
            
            // permutation board
            U64 permutation = 0;
            // add all set bits to permutation
            while (copy != 0) {
                int index = __builtin_ctzll(copy);
                permutation = permutation | posCol[index];
                copy = copy & (~(1ULL << index));
            }

            // board for attack column above the piece
            U64 col1 = 0;
            // set col1 through iteration
            for (int bit = 1; bit < 8; bit++) {
                // next column bit
                U64 stop = piece << (bit * 8);

                // prevent wraps (probably not necessary)
                if ((stop & Tables::rank1) != 0) {
                    break;
                }

                // set column bit
                col1 = col1 | stop;

                // check if blocked by piece
                U64 blocked = stop & permutation;
                if (blocked != 0) {
                    break;
                }
            }

            // board for attack column below the piece
            U64 col2 = 0;
            // set col2 through iteration
            for (int bit = 1; bit < 8; bit++) {
                // next column bit
                U64 stop = piece >> (bit * 8);

                // prevent wraps (probably not necessary)
                if ((stop & Tables::rank8) != 0) {
                    break;
                }

                // set column bit
                col2 = col2 | stop;

                // check if blocked by piece
                U64 blocked = stop & permutation;
                if (blocked != 0) {
                    break;
                }
            }

            // push all permutations and attacks
            colPerms.push_back(permutation);
            colAttacks.push_back(col1 | col2);

            // get next permutation
            colPermIndex += 1;
        }

        // get all the permutations of the rows
        while (rowPermIndex != std::pow(2, rowSize)) {
            // this index shows which row bits are set
            copy = rowPermIndex;

            // permutation board
            U64 permutation = 0;
            // add all set bits to permutation
            while (copy != 0) {
                int index = __builtin_ctzll(copy);
                permutation = permutation | posRow[index];
                copy = copy & (~(1ULL << index));
            }

            // board for attack row left of the piece
            U64 row1 = 0;
            // set row1 through iteration
            for (int bit = 1; bit < 8; bit++) {
                // next row bit
                U64 stop = piece << bit;

                // prevent wraps
                if ((stop & Tables::fileH) != 0) {
                    break;
                }

                // set row bit
                row1 = row1 | stop;

                // check if blocked by piece
                U64 blocked = stop & permutation;
                if (blocked != 0) {
                    break;
                }
            }

            // board for attack row right of the piece
            U64 row2 = 0;
            // set row2 through iteration
            for (int bit = 1; bit < 8; bit++) {
                // next row bit
                U64 stop = piece >> bit;

                // prevent wraps
                if ((stop & Tables::fileA) != 0) {
                    break;
                }

                // set row bit
                row2 = row2 | stop;

                // check if blocked by piece
                U64 blocked = stop & permutation;
                if (blocked != 0) {
                    break;
                }
            }

            // push all permutations and attacks
            rowAttacks.push_back(row1 | row2);
            rowPerms.push_back(permutation);

            // get next permutation
            rowPermIndex += 1;
        }
        
        std::vector<U64> permutations;
        std::vector<U64> attacks;

        // combine all permutations and attacks
        for (int j = 0; j < colPerms.size(); j++) {
            for (int k = 0; k < rowPerms.size(); k++) {
                permutations.push_back(colPerms[j] | rowPerms[k]);
                attacks.push_back(colAttacks[j] | rowAttacks[k]);
            }
        }

        // generate magic numbers
        U64 magicNum = generateMagicNumbers(5, permutations, attacks, rookAttacks, i);
        
        file << "0x";
        file << std::hex << magicNum;
        file << ", ";
    }
    file << "}\n";

    // output rook attacks
    file << "{{";
    for (std::vector<U64> attack : rookAttacks) {
        file << "{{";
        for (U64 board : attack) {
            file << "0x";
            file << std::hex << board;
            file << ", ";
        }
        file << "}}, ";
    }
    file << "}}";
}

// generates a vector of bishop attacks at each square
std::vector<U64> generateBishopBlockerVector() {
    std::vector<U64> bishopBlockers;

    // iterate through each square
    for (U64 square = 0; square < 64; square++) {
        U64 piece = 1ULL << square;

        // the top part of the right diagonal
        U64 rightDiagonalTop = 0;
        for (int shift = 9; shift < 64; shift += 9) {
            // shift to appropriate target
            U64 target = (piece << shift);

            // boundary checks
            if ((target & Tables::fileA) != 0 || (target & Tables::rank8) != 0 || (target & Tables::fileH) != 0 || target == 0) {
                break;
            }
            rightDiagonalTop = rightDiagonalTop | target;
        }

        // the bottom part of the right diagonal
        U64 rightDiagonalBottom = 0;
        for (int shift = 9; shift < 64; shift += 9) {
            // shift to appropriate target
            U64 target = (piece >> shift);

            // boundary checks
            if ((target & Tables::fileA) != 0 || (target & Tables::rank1) != 0 || (target & Tables::fileH) != 0 || target == 0) {
                break;
            }
            rightDiagonalBottom = rightDiagonalBottom | target;
        }

        // the top part of the left diagonal
        U64 leftDiagonalTop = 0;
        for (int shift = 7; shift < 64; shift += 7) {
            // shift to appropriate target
            U64 target = (piece << shift);

            // boundary checks
            if ((target & Tables::fileA) != 0 || (target & Tables::rank8) != 0 || (target & Tables::fileH) != 0 || target == 0) {
                break;
            }
            leftDiagonalTop = leftDiagonalTop | target;
        }

        // the bottom part of the left diagonal
        U64 leftDiagonalBottom = 0;
        for (int shift = 7; shift < 64; shift += 7) {
            // shift to appropriate target
            U64 target = (piece >> shift);

            // boundary checks
            if ((target & Tables::fileA) != 0 || (target & Tables::rank1) != 0 || (target & Tables::fileH) != 0 || target == 0) {
                break;
            }
            leftDiagonalBottom = leftDiagonalBottom | target;
        }

        // merge and push
        bishopBlockers.push_back(rightDiagonalBottom | rightDiagonalTop | leftDiagonalBottom | leftDiagonalTop);
    }

    return bishopBlockers;
}

// generate pieces able to block bishop and outputs to file
void generateBishopBlockers(std::ofstream& file) {
    // file: output file

    file << "{";

    // get blockers
    std::vector<U64> bishopBlockers = generateBishopBlockerVector();

    // output each square
    for (U64 square : bishopBlockers) {
        file << "0x";
        file << std::hex << square;
        file << ", ";
    }

    file << "}";
}

// generate how many moves bishop is able to get to at each square
void generateBishopMagicNumberLengths(std::ofstream& file) {
    // file: output file

    file << "{";

    // get blockers
    std::vector<U64> bishopBlockers = generateBishopBlockerVector();

    // output each square
    for (U64 square : bishopBlockers) {
        U64 copy = square;
        int count = 0;

        while (copy != 0) {
            U64 index = __builtin_ctzll(copy);
            copy = copy & (~(1ULL << index));
            count += 1;
        }

        file << count;
        file << ", ";
    }

    file << "}";
}

// generate bishop magic numbers and bishop attack tables
void generateBishopMagics(std::ofstream& file) {
    // file: output file

    std::vector<U64> bishopBlockers = generateBishopBlockerVector();
    std::vector<std::vector<U64>> bishopAttacks;

    file << "{";

    for (int i = 0; i < 64; i++) {
        U64 piece = (1ULL << i);
        std::vector<U64> squarePositions;
        U64 copy = bishopBlockers[i];

        while (copy != 0) {
            int index = __builtin_ctzll(copy);
            squarePositions.push_back(index);
            copy = copy & ~(1ULL << index);
        }

        U64 indices = 0;
        std::vector<U64> permutations;
        std::vector<U64> attacks;
        
        while (indices < (1ULL << squarePositions.size())) {
            copy = indices;
            U64 permutation = 0;

            while (copy != 0) {
                int index = __builtin_ctzll(copy);
                permutation = permutation | (1ULL << squarePositions[index]);
                copy = copy & ~(1ULL << index);
            }

            permutations.push_back(permutation);

            // the top part of the right diagonal
            U64 rightDiagonalTop = 0;
            for (int shift = 9; shift < 64; shift += 9) {
                // shift to appropriate target
                U64 target = (piece << shift);

                // boundary checks
                if ((target & Tables::fileH) != 0 || (target & Tables::rank1) != 0) {
                    break;
                }
                rightDiagonalTop = rightDiagonalTop | target;

                // check for blocker
                if ((target & permutation) != 0) {
                    break;
                }
            }

            // the bottom part of the right diagonal
            U64 rightDiagonalBottom = 0;
            for (int shift = 9; shift < 64; shift += 9) {
                // shift to appropriate target
                U64 target = (piece >> shift);

                // boundary checks
                if ((target & Tables::fileA) != 0 || (target & Tables::rank8) != 0) {
                    break;
                }
                rightDiagonalBottom = rightDiagonalBottom | target;

                // check for blocker
                if ((target & permutation) != 0) {
                    break;
                }
            }

            // the top part of the left diagonal
            U64 leftDiagonalTop = 0;
            for (int shift = 7; shift < 64; shift += 7) {
                // shift to appropriate target
                U64 target = (piece << shift);

                // boundary checks
                if ((target & Tables::fileA) != 0 || (target & Tables::rank1) != 0) {
                    break;
                }
                leftDiagonalTop = leftDiagonalTop | target;

                // check for blocker
                if ((target & permutation) != 0) {
                    break;
                }
            }

            // the bottom part of the left diagonal
            U64 leftDiagonalBottom = 0;
            for (int shift = 7; shift < 64; shift += 7) {
                // shift to appropriate target
                U64 target = (piece >> shift);

                // boundary checks
                if ((target & Tables::fileH) != 0 || (target & Tables::rank8) != 0) {
                    break;
                }
                leftDiagonalBottom = leftDiagonalBottom | target;

                // check for blocker
                if ((target & permutation) != 0) {
                    break;
                }
            }

            attacks.push_back(rightDiagonalBottom | rightDiagonalTop | leftDiagonalBottom | leftDiagonalTop);
            indices += 1;
        }

        // generate magic numbers
        U64 magicNum = generateMagicNumbers(5, permutations, attacks, bishopAttacks, i);
        
        file << "0x";
        file << std::hex << magicNum;
        file << ", ";
    }
    file << "}\n";

    // output rook attacks
    file << "{{";
    for (std::vector<U64> attack : bishopAttacks) {
        file << "{{";
        for (U64 board : attack) {
            file << "0x";
            file << std::hex << board;
            file << ", ";
        }
        file << "}}, ";
    }
    file << "}}";
}

// generate the pinmask table given two squares
void generateRookPinMasks(std::ofstream& file) {
    // file: output file

    file << "{{";

    // iterate through all possible two square combinations
    for (int first = 0; first < 64; first++) {
        file << "{{";

        // first piece bitboard
        U64 piece1 = 1ULL << first;

        // masks
        U64 currentRank = Tables::rank1 << ((first / 8) * 8);
        U64 currentFile = Tables::fileH << (first % 8);

        for (int second = 0; second < 64; second++) {
            // second piece bitboard
            U64 piece2 = 1ULL << second;

            // ignore same square combinations
            if (piece1 == piece2) {
                file << "0x0, ";
            } else {
                // mask piece to check rook rays
                U64 sameRank = currentRank & piece2;
                U64 sameFile = currentFile & piece2;

                // check if pin is possible using masked variables
                if (sameRank != 0) {
                    file << "0x";
                    file << std::hex << currentRank;
                    file << ", ";
                } else if (sameFile != 0) {
                    file << "0x";
                    file << std::hex << currentFile;
                    file << ", ";
                } else {
                    file << "0x0, ";
                }
            }
        }
        file << "}}, ";
    }
    file << "}}";
}

// generate common attacks between two rooks
void generateRookTwoPieceAttacks(std::ofstream& file) {
    // file: output file

    file << "{{";

    // iterate through all possible two square combinations
    for (int first = 0; first < 64; first++) {
        file << "{{";

        // first piece bitboard
        U64 piece1 = 1ULL << first;

        // masks
        U64 currentRank = Tables::rank1 << ((first / 8) * 8);
        U64 currentFile = Tables::fileH << (first % 8);

        for (int second = 0; second < 64; second++) {
            // second piece bitboard
            U64 piece2 = 1ULL << second;

            // ignore same square combinations
            if (first == second) {
                file << "0x0, ";
            } else {
                // mask pieces to check rook rays
                U64 sameRank = currentRank & piece2;
                U64 sameFile = currentFile & piece2;

                // get the common rook attack ray of both pieces
                U64 piece1Targets = Position::rookAttacks(piece2, first);
                U64 piece2Targets = Position::rookAttacks(piece1, second);
                U64 combined = piece1Targets & piece2Targets;
                
                // check if pieces are on the same rank or file
                if (sameRank != 0 || sameFile != 0) {
                    file << "0x";
                    file << std::hex << combined;
                    file << ", ";
                } else {
                    file << "0x0, ";
                }
            }
        }
        file << "}}, ";
    }
    file << "}}";
}

// get the diagonals where a bishop can pin
std::vector<U64> generateBishopDiagonals() {
    // output: vector of bishop diagonals

    std::vector<U64> bishopDiagonals;

    // ignore 1 or 2 square diagonals, since they cannot have pinned pieces
    for (int index = 2; index < 8; index++) {
        // mask diagonal with inverted diagonal to get intended diagonal
        // bottom right diagonals
        bishopDiagonals.push_back(Position::bishopAttacks(Tables::empty, 8 * index) 
                                & Position::bishopAttacks(Tables::empty, index)
                                | (1ULL << index) | (1ULL << (index * 8)));
        // top left diagonals
        bishopDiagonals.push_back(Position::bishopAttacks(Tables::empty, 8 * (8 - index) - 1)
                                & Position::bishopAttacks(Tables::empty, 63 - index)
                                | (1ULL << (63 - index)) | (1ULL << (63 - 8 * index)));
        // bottom left diagonals
        bishopDiagonals.push_back(Position::bishopAttacks(Tables::empty, 8 * index + 7)
                                & Position::bishopAttacks(Tables::empty, 7 - index)
                                | (1ULL << (7 - index)) | (1ULL << (index * 8 + 7)));
        // top right diagonals
        bishopDiagonals.push_back(Position::bishopAttacks(Tables::empty, 56 + index)
                                & Position::bishopAttacks(Tables::empty, 8 * (7 - index))
                                | (1ULL << (56 + index)) | (1ULL << (56 - index * 8)));
    }

    return bishopDiagonals;
}

// generate common attacks between two bishops
void generateBishopPinmasks(std::ofstream& file) {
    // file: output file

    file << "{{";

    // get bishop diagonals
    std::vector<U64> diagonals = generateBishopDiagonals();

    // iterate through all two square combinations
    for (int first = 0; first < 64; first++) {
        U64 piece1 = 1ULL << first;
        file << "{{";

        for (int second = 0; second < 64; second++) {
            U64 piece2 = 1ULL << second;
            bool success = false;

            // check if there is a possible pin
            for (U64 diagonal : diagonals) {
                // if pieces lie on the same diagonal, one is pinned
                if (__builtin_popcountll(diagonal & piece1) == 1 && __builtin_popcountll(diagonal & piece2) == 1) {
                    // output pinmask and set success flag
                    file << "0x";
                    file << std::hex << diagonal;
                    file << ", ";
                    success = true;
                    break;
                }
            }

            if (!success) {
                file << "0x0, ";
            }
        }
        file << "}}, ";
    }
    file << "}}";
}

// generate a table of bishop 
void generateBishopTwoPieceAttacks(std::ofstream& file) {
    // file: output file

    file << "{{";

    // get bishop diagonals
    std::vector<U64> diagonals = generateBishopDiagonals();

    // iterate through all two square combinations
    for (int first = 0; first < 64; first++) {
        U64 piece1 = 1ULL << first;
        file << "{{";

        for (int second = 0; second < 64; second++) {
            U64 piece2 = 1ULL << second;
            bool success = false;

            // check if there is a possible pin
            for (U64 diagonal : diagonals) {
                // if pieces lie on the same diagonal, one is pinned
                if (__builtin_popcountll(diagonal & piece1) == 1 && __builtin_popcountll(diagonal & piece2) == 1) {
                    // output pinmask and set success flag
                    file << "0x";
                    file << std::hex << (Position::bishopAttacks(piece2, first) & Position::bishopAttacks(piece1, second));
                    file << ", ";
                    success = true;
                    break;
                }
            }

            if (!success) {
                file << "0x0, ";
            }
        }
        file << "}}, ";
    }
    file << "}}";
}

int main() {
    // initialize and open file
    std::ofstream file;
    file.open("./build/generatedTables.txt", std::ios_base::app);

    // generate tables
    generateKnightBitboards(file);
    file << "\n";
    generateKingBitboards(file);
    file << "\n";
    generateRookBlockerBitboards(file);
    file << "\n";
    generateRookMagics(file);
    file << "\n";
    generateBishopBlockers(file);
    file << "\n";
    generateBishopMagicNumberLengths(file);
    file << "\n";
    generateBishopMagics(file);
    file << "\n";
    generateRookPinMasks(file);
    file << "\n";
    generateRookTwoPieceAttacks(file);
    file << "\n";
    generateBishopPinmasks(file);
    file << "\n";
    generateBishopTwoPieceAttacks(file);
    file.close();
}
