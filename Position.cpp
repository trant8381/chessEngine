// Bitboards.cpp
// Bitboard specific definitions

#include "Position.h"
#include "Tables.h"
#include "Types.h"
#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <x86intrin.h>
#include <ostream>
#include <set>

// debugging function to print U64 to binary
void printBoard(U64 board) {
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
		std::cout << binary[i];
		if ((i + 1) % 8 == 0) {
			std::cout << std::endl;


		}
	}

	std::cout << std::endl;
}

// void printBoard(U64& board) {
//     /*
//         board: bitboard to display
//         file: output file
//     */
// 	std::ofstream file;
// 	file.open("./build/generatedTables.txt", std::ios_base::app);
//     std::string binary = "";
//     U64 temp = board;

//     // number to binary
//     while (temp != 0) {
//         int mod = temp % 2;
//         binary += std::to_string(mod);
//         temp = temp / 2;
//     }

//     // add some zeroes
//     int end = 64 - binary.length();
//     for (int i = 0; i < end; i++) {
//         binary += "0";
//     }

//     std::reverse(binary.begin(), binary.end());

//     // print board
//     for (int i = 0; i < 64; i++) {
//         file << binary[i];
//         if ((i + 1) % 8 == 0) {
//             file << std::endl;
//         }
//     }

//     file << std::endl;
// 	file.close();
// }

// flips least significant bit and returns index
int lsb(U64& board) {
	int index = __builtin_ctzll(board); // count zeroes up to first true bit
	board = __blsr_u64(board);		    // remove bit

	return index;
}

// pushes all moves from bitboard into moveset
void pushMoves(U64& targets, const int& index, Moveset &moveset) {
	// iterates until all targets are removed
	int targetsPopCount = __builtin_popcountll(targets);
	for (int _ = 0; _ < targetsPopCount; _++) {
		moveset.normal.push_back({index, lsb(targets)});
	}
}

// gets the attack bitboard of one white pawn
U64 Position::whitePawnAttacks(const U64& pawn) {
	// pawn right capture bitboard
	U64 rightCapture = ((pawn & Tables::notFileH) << 7);

	// pawn left capture bitboard
	U64 leftCapture = ((pawn & Tables::notFileA) << 9);

	return rightCapture | leftCapture;
}


// get all white pawn moves and push to moves vector
void Position::whitePawnMoves(Moveset &moveset) {
	// get empty squares and exclude promoting pawns
	U64 emptySquares = ~pieces;
	U64 notPromoting = whitePawns & Tables::notRank7;
	int popCount = __builtin_popcountll(notPromoting);

	for (int _ = 0; _ < popCount; _++) {
		int index = lsb(notPromoting);
		U64 pawn = 1ULL << index;
		U64 up = (pawn << 8) & emptySquares;
		U64 doubleUp = ((up & Tables::rank3) << 8) & emptySquares & checkMask & grid[index].pinMask;
		U64 rightAttack = ((pawn & Tables::notFileH) << 7);
		U64 leftAttack = ((pawn & Tables::notFileA) << 9);
		U64 rightCapture = rightAttack & blackPieces;
		U64 leftCapture = leftAttack & blackPieces;
		U64 canEnPassant = (leftAttack & enPassant) | (rightAttack & enPassant);
		U64 normalTargets = (up | rightCapture | leftCapture) & checkMask & grid[index].pinMask;

		pushMoves(normalTargets, index, moveset);

		if (doubleUp != 0) {
			moveset.doubleMoves.push_back({index, __builtin_ctzll(doubleUp), __builtin_ctzll(up)});
		}

		if (canEnPassant != 0) {
			moveset.enPassant.push_back({index, __builtin_ctzll(canEnPassant)});
		}
	}
}

// gets the attack bitboard of one black pawn
U64 Position::blackPawnAttacks(const U64& pawn) {
	// pawn right capture bitboard
	U64 rightCapture = ((pawn & Tables::notFileH) >> 9);

	// pawn left capture bitboard
	U64 leftCapture = ((pawn & Tables::notFileA) >> 7);
	return rightCapture | leftCapture;
}

// get all black pawn moves and push to moves vector
void Position::blackPawnMoves(Moveset &moveset) {
	// get empty squares and exclude promoting pawns
	U64 emptySquares = ~pieces;
	U64 notPromoting = blackPawns & Tables::notRank2;
	int popCount = __builtin_popcountll(notPromoting);

	for (int _ = 0; _ < popCount; _++) {
		int index = lsb(notPromoting);
		U64 pawn = 1ULL << index;
		U64 up = (pawn >> 8) & emptySquares;
		U64 doubleUp = ((up & Tables::rank6) >> 8) & emptySquares & checkMask & grid[index].pinMask;
		U64 rightAttack = ((pawn & Tables::notFileH) >> 9);
		U64 leftAttack = ((pawn & Tables::notFileA) >> 7);
		U64 rightCapture = rightAttack & whitePieces;
		U64 leftCapture = leftAttack & whitePieces;
		U64 canEnPassant = (leftAttack & enPassant) | (rightAttack & enPassant);
		U64 normalTargets = (up | rightCapture | leftCapture) & checkMask & grid[index].pinMask;
		pushMoves(normalTargets, index, moveset);

		if (doubleUp != 0) {
			moveset.doubleMoves.push_back({index, __builtin_ctzll(doubleUp), __builtin_ctzll(up)});
		}
		if (canEnPassant != 0) {
			moveset.enPassant.push_back({index, __builtin_ctzll(canEnPassant)});
		}
	}
	
}

// gets the attack bitboard of one knight
U64 Position::knightAttacks(const int& index) {
	return Tables::knightBitboards[index];
}

// gets knight moves from table and push to moves vector
void Position::knightMoves(Moveset &moveset, const U64& own, U64 knights) {
	// gets all knights and pushes their moves
	int size = __builtin_popcountll(knights);

	for (int i = 0; i < size; i++) {
		int index = lsb(knights);
		
		// get bitboard using index and mask with pieces
		U64 targets = knightAttacks(index) & (~own) & checkMask & grid[index].pinMask;
		pushMoves(targets, index, moveset);
	}
}

// gets the attack bitboard of one rook
U64 Position::rookAttacks(const U64& blockMask, const int& index) {
	// get rook targets through table
	U64 targets = (blockMask & Tables::rookBlockerBitboards[index]);
	targets *= Tables::rookMagicNumbers[index];
	// shift relevant index bits down
	targets = targets >> (64 - Tables::rookMagicNumberLengths[index]);
	// get attack targets
	targets = Tables::rookAttacks[index][targets];

	return targets;
}

// xray rook attacks
U64 Position::xrayRookAttacks(const U64& occ, const U64& blockers, const int& index) {
   U64 attacks = rookAttacks(occ, index);
   U64 between = occ ^ (blockers & attacks);
   return attacks ^ rookAttacks(between, index);
}

// gets rook moves from table and push to moves vector
void Position::rookMoves(Moveset &moveset, const U64 &own, U64 rooks) {
	// gets all rooks and pushes their moves
	int size = __builtin_popcountll(rooks);

	for (int i = 0; i < size; i++) {
		int index = lsb(rooks);
		U64 targets = rookAttacks(pieces, index) & (~own) & checkMask & grid[index].pinMask;
		pushMoves(targets, index, moveset);
	}
}

// gets the attack bitboard of one bishop
U64 Position::bishopAttacks(const U64& blockMask, const int& index) {
	// get bishop targets through table
	U64 targets = (blockMask & Tables::bishopBlockerBitboards[index]);
	targets *= Tables::bishopMagicNumbers[index];
	// shift relevant index bits down
	targets = targets >> (64 - Tables::bishopMagicNumberLengths[index]);
	// get attack targets
	targets = Tables::bishopAttacks[index][targets];

	return targets;
}

// xray bishop attacks
U64 Position::xrayBishopAttacks(const U64& occ, const U64& blockers, const int& index) {
   U64 attacks = bishopAttacks(occ, index);
   U64 between = occ ^ (blockers & attacks);
   return attacks ^ bishopAttacks(between, index);
}

// gets bishop moves from table and push to moves vector
void Position::bishopMoves(Moveset &moveset, const U64 &own, U64 bishops) {
	// gets all bishops and pushes their moves
	int size = __builtin_popcountll(bishops);
	for (int i = 0; i < size; i++) {
		int index = lsb(bishops);
		U64 targets = bishopAttacks(pieces, index) & (~own) & checkMask & grid[index].pinMask;
		pushMoves(targets, index, moveset);
	}
}

U64 Position::queenAttacks(U64& blockMask, int& index) {
	return bishopAttacks(blockMask, index) | rookAttacks(blockMask, index);
}

// gets queen moves from rook and bishop functions and push to moves vector
void Position::queenMoves(Moveset &moveset, const U64 &own, U64& queens) {
	rookMoves(moveset, own, queens);
	bishopMoves(moveset, own, queens);
}

U64 Position::kingAttacks(const int& index) {
	return Tables::kingBitboards[index];
}

// gets black king moves from table and push to moves vector
// returns true if there is a double check on the king, false otherwise
bool Position::blackMoves(Moveset& moveset) {
	// moves: vector of moves
	// output: true for double check, false otherwise

	U64 copy = blackKing;
	int index = lsb(copy);

    // get bitboard using index and mask
	U64 targets = Tables::kingBitboards[index] & (~blackPieces);
	U64 attacks = 0;
	U64 blockMask = pieces & (~blackKing);

	copy = whiteBishops;
	int size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | bishopAttacks(blockMask, lsb(copy));
	}

	copy = whiteRooks;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | rookAttacks(blockMask, lsb(copy));
	}

	copy = whiteQueens;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		int queen = lsb(copy);
		attacks = attacks | queenAttacks(blockMask, queen);
	}

	copy = whiteKnights;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | knightAttacks(lsb(copy));
	}

	copy = whitePawns;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		U64 pawn = 1ULL << lsb(copy);
		attacks = attacks | whitePawnAttacks(pawn);
	}

	copy = whiteKing;
	int king = lsb(copy);
	attacks = attacks | kingAttacks(king);
	targets = targets & (~attacks);

	pushMoves(targets, index, moveset);

	if ((!grid[59].hasMoved) && (!grid[56].hasMoved) && ((pieces | attacks) & 0x0600000000000000) == 0) {
		moveset.castle.push_back({{{{59, 57}}, {{56, 58}}}});
	}

	if ((!grid[59].hasMoved) && (!grid[63].hasMoved) && ((pieces | attacks) & 0x3000000000000000) == 0) {
		moveset.castle.push_back({{{{59, 61}}, {{63, 60}}}});
	}

	// get pinned pieces
	U64 rookPinners = rookAttacks(0, index) & (whiteQueens | whiteRooks);
	int rookPinnerPopCount = __builtin_popcountll(rookPinners); // get amount of set bits

	// iterate through all rook pinners
	for (int _ = 0; _ < rookPinnerPopCount; _++) {
		int pinner = lsb(rookPinners);
		// get all white pieces between king and rook
		U64 possiblyPinned = Tables::rookTwoPieceAttacks[index][pinner] & pieces;

		// if there is one piece, it is pinned
		if (__builtin_popcountll(possiblyPinned) == 1 && (possiblyPinned & blackPieces) != 0) {
			int pinned = __builtin_ctzll(possiblyPinned);

			// record pinMask
			grid[pinned].pinMask = Tables::rookPinMasks[pinned][pinner];
		}
	}

	U64 bishopPinners = bishopAttacks(0, index) & (whiteQueens | whiteBishops);
	int bishopPinnerPopCount = __builtin_popcountll(bishopPinners); // get amount of set bits

	// iterate through all rook pinners
	for (int _ = 0; _ < bishopPinnerPopCount; _++) {
		int pinner = lsb(bishopPinners);
		// get all white pieces between king and rook
		U64 possiblyPinned = Tables::bishopTwoPieceAttacks[index][pinner] & pieces;

		// if there is one piece, it is pinned
		if (__builtin_popcountll(possiblyPinned) == 1 && (possiblyPinned & blackPieces) != 0) {
			int pinned = __builtin_ctzll(possiblyPinned);

			// record pinMask
			grid[pinned].pinMask = Tables::bishopPinMasks[pinned][pinner];
		}
	}

	if ((attacks & blackKing) == 0) {
		checkMask = 0xFFFFFFFFFFFFFFFF;
		bishopMoves(moveset, blackPieces, blackBishops);
		rookMoves(moveset, blackPieces, blackRooks);
		queenMoves(moveset, blackPieces, blackQueens);
		knightMoves(moveset, blackPieces, blackKnights);
		blackPawnMoves(moveset);

		return 0;
	}
	
	// define new checkmask
	checkMask = 0;

	U64 rookTargets = rookAttacks(pieces, index);
    U64 rookChecker = rookTargets & (whiteQueens | whiteRooks);
	int rookCheck = false;

	if (rookChecker != 0) {
		rookCheck = true;
		int rook = __builtin_ctzll(rookChecker);
		U64 reversedAttack = rookAttacks(pieces, rook);
		checkMask = checkMask | (rookTargets & reversedAttack) | rookChecker;
	}
    
	U64 bishopTargets = bishopAttacks(pieces, index);
	U64 bishopChecker = bishopTargets & (whiteQueens | whiteBishops);
    int bishopCheck = false;

	if (bishopChecker != 0) {
		bishopCheck = true;
		int bishop = __builtin_ctzll(bishopChecker);
		U64 reversedAttack = bishopAttacks(pieces, bishop);
		checkMask = checkMask | (bishopTargets & reversedAttack) | bishopChecker;
	}

	U64 knightTargets = knightAttacks(index);
	U64 knightChecker = knightTargets & whiteKnights;
	int knightCheck = knightChecker >> __builtin_ctzll(knightChecker);
	checkMask = checkMask | (knightChecker);

	U64 pawn = 1ULL << index;
	U64 pawnTargets = blackPawnAttacks(pawn);
	U64 pawnChecker = pawnTargets & whitePawns;
	int pawnCheck = pawnChecker >> __builtin_ctzll(pawnChecker);
	checkMask = checkMask | (pawnChecker);

	if (rookCheck + bishopCheck + knightCheck + pawnCheck == 2) {
		return 1;
	}

	bishopMoves(moveset, blackPieces, blackBishops);
	rookMoves(moveset, blackPieces, blackRooks);
	queenMoves(moveset, blackPieces, blackQueens);
	knightMoves(moveset, blackPieces, blackKnights);
	blackPawnMoves(moveset);

	return 0;
}

// gets possible white moves, initializes pinmasks and checkmasks
bool Position::whiteMoves(Moveset& moveset) {
	// moves: vector of moves
	// output: true for double check, false otherwise

	U64 copy = whiteKing;
	int index = lsb(copy);

    // get bitboard using index and mask
	U64 targets = Tables::kingBitboards[index] & ~whitePieces;
	U64 attacks = 0;
	U64 blockMask = pieces & (~whiteKing);

	copy = blackBishops;
	int size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | bishopAttacks(blockMask, lsb(copy));
	}

	copy = blackRooks;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | rookAttacks(blockMask, lsb(copy));
	}

	copy = blackQueens;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		int queen = lsb(copy);
		attacks = attacks | queenAttacks(blockMask, queen);
	}

	copy = blackKnights;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		attacks = attacks | knightAttacks(lsb(copy));
	}

	copy = blackPawns;
	size = __builtin_popcountll(copy);
	for (int i = 0; i < size; i++) {
		U64 pawn = 1ULL << lsb(copy);
		attacks = attacks | blackPawnAttacks(pawn);
	}

	copy = blackKing;
	int king = lsb(copy);
	attacks = attacks | kingAttacks(king);
	targets = targets & (~attacks);

	pushMoves(targets, index, moveset);
	
	if ((!grid[3].hasMoved) && (!grid[0].hasMoved) && ((pieces | attacks) & 6) == 0) {
		moveset.castle.push_back({{{{3, 1}}, {{0, 2}}}});
	}

	if ((!grid[3].hasMoved) && (!grid[7].hasMoved) && ((pieces | attacks) & 48) == 0) {
		moveset.castle.push_back({{{{3, 5}}, {{7, 4}}}});
	}

	// apply pinmasks to pinned pieces
	U64 rookPinners = rookAttacks(0, index) & (blackQueens | blackRooks);
	int rookPinnerPopCount = __builtin_popcountll(rookPinners); // get amount of set bits

	// iterate through all rook pinners
	for (int _ = 0; _ < rookPinnerPopCount; _++) {
		int pinner = lsb(rookPinners);
		// get all white pieces between king and rook
		U64 possiblyPinned = Tables::rookTwoPieceAttacks[index][pinner] & pieces;

		// if there is one piece, it is pinned
		if (__builtin_popcountll(possiblyPinned) == 1 && (possiblyPinned & whitePieces) != 0) {
			int pinned = __builtin_ctzll(possiblyPinned);

			// record pinMask
			grid[pinned].pinMask = Tables::rookPinMasks[pinned][pinner];
		}
	}

	U64 bishopPinners = bishopAttacks(0, index) & (blackQueens | blackBishops);
	int bishopPinnerPopCount = __builtin_popcountll(bishopPinners); // get amount of set bits

	// iterate through all rook pinners
	for (int _ = 0; _ < bishopPinnerPopCount; _++) {
		int pinner = lsb(bishopPinners);
		// get all white pieces between king and rook
		U64 possiblyPinned = Tables::bishopTwoPieceAttacks[index][pinner] & pieces;

		// if there is one piece, it is pinned
		if (__builtin_popcountll(possiblyPinned) == 1 && (possiblyPinned & whitePieces) != 0) {
			int pinned = __builtin_ctzll(possiblyPinned);

			// record pinMask
			grid[pinned].pinMask = Tables::bishopPinMasks[pinned][pinner];
		}
	}

	if ((attacks & whiteKing) == 0) {
		checkMask = Tables::full;
		bishopMoves(moveset, whitePieces, whiteBishops);
		rookMoves(moveset, whitePieces, whiteRooks);
		queenMoves(moveset, whitePieces, whiteQueens);
		knightMoves(moveset, whitePieces, whiteKnights);
		whitePawnMoves(moveset);

		return 0;
	}
	
	// define new checkmask
	checkMask = 0;

	U64 rookTargets = rookAttacks(pieces, index);
    U64 rookChecker = rookTargets & (blackQueens | blackRooks);
	int rookCheck = false;

	if (rookChecker != 0) {
		rookCheck = true;
		int rook = __builtin_ctzll(rookChecker);
		U64 reversedAttack = rookAttacks(pieces, rook);
		checkMask = checkMask | (rookTargets & reversedAttack) | rookChecker;
	}
    
	U64 bishopTargets = bishopAttacks(pieces, index);
	U64 bishopChecker = bishopTargets & (blackQueens | blackBishops);
    int bishopCheck = false;

	if (bishopChecker != 0) {
		bishopCheck = true;
		int bishop = __builtin_ctzll(bishopChecker);
		U64 reversedAttack = bishopAttacks(pieces, bishop);
		checkMask = checkMask | (bishopTargets & reversedAttack) | bishopChecker;
	}

	U64 knightTargets = knightAttacks(index);
	U64 knightChecker = knightTargets & blackKnights;
	int knightCheck = knightChecker >> __builtin_ctzll(knightChecker);
	checkMask = checkMask | (knightChecker);

	U64 pawn = 1ULL << index;
	U64 pawnTargets = whitePawnAttacks(pawn);
	U64 pawnChecker = pawnTargets & blackPawns;
	int pawnCheck = pawnChecker >> __builtin_ctzll(pawnChecker);
	checkMask = checkMask | (pawnChecker);
	
	// if double check, there is no need to enumerate non king moves
	if (rookCheck + bishopCheck + knightCheck + pawnCheck == 2) {
		return 1;
	}

	// get other moves
	bishopMoves(moveset, whitePieces, whiteBishops);
	rookMoves(moveset, whitePieces, whiteRooks);
	queenMoves(moveset, whitePieces, whiteQueens);
	knightMoves(moveset, whitePieces, whiteKnights);
	whitePawnMoves(moveset);

	return 0;
}

Position Position::copy() {
	Position newPosition;

	// make a copy of everything
	newPosition.whitePawns = whitePawns;
	newPosition.blackPawns = blackPawns;
	newPosition.blackRooks = blackRooks;
	newPosition.whiteRooks = whiteRooks;
	newPosition.blackKnights = blackKnights;
	newPosition.whiteKnights = whiteKnights;
	newPosition.blackBishops = blackBishops;
	newPosition.whiteBishops = whiteBishops;
	newPosition.blackKing = blackKing;
	newPosition.whiteKing = whiteKing;
	newPosition.blackQueens = blackQueens;
	newPosition.whiteQueens = whiteQueens;
	newPosition.whitePieces = whitePieces;
	newPosition.blackPieces = blackPieces;
	newPosition.pieces = pieces;
	newPosition.whiteQueenCastle = whiteQueenCastle;
	newPosition.whiteKingCastle = whiteKingCastle;
	newPosition.blackQueenCastle = blackQueenCastle;
	newPosition.blackKingCastle = blackKingCastle;
	newPosition.isWhiteTurn = !isWhiteTurn;
	newPosition.moveCount = moveCount + 1;
	newPosition.grid = {};

	// reset the pin mask
	for (int i = 0; i < 64; i++) {
		Piece piece = grid[i];
		newPosition.grid[i] = piece;
		newPosition.grid[i].pinMask = 0xffffffffffffffff;
	}

	return newPosition;
}

Position Position::makeNormalMove(int& start, int& end) {
	Position newPosition = copy();
	Piece piece = grid[start];
	piece.hasMoved = true;
	Piece capturedPiece = grid[end];

	newPosition.grid[start] = Piece(15, 15);
	newPosition.grid[end] = piece;

	*newPosition.bitboardPointers[capturedPiece.bitboardIndex] &= ~(1Ull << end);
	*newPosition.bitboardPointers[capturedPiece.colorBitboardIndex] &= ~(1ULL << end);

	U64& board = *newPosition.bitboardPointers[piece.bitboardIndex];
 	U64 notStart = ~(1ULL << start);
	U64 endBoard = (1ULL << end);
	board = (board & notStart) | endBoard;
	U64& colorBoard = *newPosition.bitboardPointers[piece.colorBitboardIndex];
	colorBoard = (colorBoard & notStart) | endBoard;
	newPosition.pieces = (pieces & notStart) | endBoard;

	return newPosition;
}

Position Position::makeCastlingMove(int &start1, int &end1, int &start2, int &end2) {
	Position newPosition = copy();
	Piece piece1 = grid[start1];
	Piece piece2 = grid[start2];

	piece1.hasMoved = true;
	piece2.hasMoved = true;

	newPosition.grid[start1] = Piece(15, 15);
	newPosition.grid[end1] = piece1;
	newPosition.grid[start2] = Piece(15, 15);
	newPosition.grid[end2] = piece2;

	U64& board1 = *newPosition.bitboardPointers[piece1.bitboardIndex];
	U64& board2 = *newPosition.bitboardPointers[piece2.bitboardIndex];

	U64 notStart1 = ~(1ULL << start1);
	U64 endBoard1 = (1ULL << end1);
	U64 notStart2 = ~(1ULL << start2);
	U64 endBoard2 = (1ULL << end2);

	board1 = (board1 & notStart1) | endBoard1;
	board2 = (board2 & notStart2) | endBoard2;

	U64& colorBoard = *newPosition.bitboardPointers[piece1.colorBitboardIndex];
	colorBoard = (colorBoard & notStart1 & notStart2) | endBoard1 | endBoard2;
	newPosition.pieces = (pieces & notStart1 & notStart2) | endBoard1 | endBoard2;

	return newPosition;
}

Position Position::makeEnPassantMove(int &start, int &end) {
	Position newPosition = copy();

	Piece piece = grid[start];
	Piece& capturedPiece = newPosition.grid[__builtin_ctzll(enPassant)];

	piece.hasMoved = true;
	newPosition.grid[start] = Piece(15, 15);
	newPosition.grid[end] = piece;

	*newPosition.bitboardPointers[capturedPiece.bitboardIndex] &= ~(1Ull << end);
	*newPosition.bitboardPointers[capturedPiece.colorBitboardIndex] &= ~(1ULL << end);
	capturedPiece = Piece(15, 15);

	U64& board = *newPosition.bitboardPointers[piece.bitboardIndex];
	U64 notStart = ~(1ULL << start);
	U64 notEnPassant = ~enPassant;
	U64 endBoard = (1ULL << end);
	board = (board & notStart & notEnPassant) | endBoard;
	U64& colorBoard = *newPosition.bitboardPointers[piece.colorBitboardIndex];
	colorBoard = (colorBoard & notStart & notEnPassant) | endBoard;
	newPosition.pieces = (pieces & notStart & notEnPassant) | endBoard;
	newPosition.enPassant = 0;

	return newPosition;
}

Position Position::makeDoubleMove(int& start, int& end, int& canEnPassant) {
	Position newPosition = copy();

	Piece piece = newPosition.grid[start];
	piece.hasMoved = true;

	newPosition.grid[start] = Piece(15, 15);
	newPosition.grid[end] = piece;

	U64& board = *newPosition.bitboardPointers[piece.bitboardIndex];
	U64 notStart = ~(1ULL << start);
	U64 endBoard = (1ULL << end);
	board = (board & notStart) | endBoard;
	U64& colorBoard = *newPosition.bitboardPointers[piece.colorBitboardIndex];
	colorBoard = (colorBoard & notStart) | endBoard;
	newPosition.pieces = (pieces & notStart) | endBoard;
	newPosition.enPassant = 1ULL << canEnPassant;

	return newPosition;
}

U64 Position::perft(int depth, std::stack<Position>& movelist) {

	Moveset moveset;
	
	U64 nodes = 0;
	if (isWhiteTurn) {
		whiteMoves(moveset);
	} else {
		blackMoves(moveset);
	}

	if (depth == 1) {		
		return moveset.normal.size + moveset.castle.size + moveset.doubleMoves.size + moveset.enPassant.size;
	}

	Array<std::array<int, 2>, 218>& normalMoves = moveset.normal;
	int& normalSize = normalMoves.size;
	for (int move = 0; move < normalSize; move++) {
		movelist.push(makeNormalMove(normalMoves[move][0], normalMoves[move][1]));
		nodes += movelist.top().perft(depth - 1, movelist);
		movelist.pop();
	}

	Array<std::array<std::array<int, 2>, 2>, 2>& castleMoves = moveset.castle;
	int& castleSize = castleMoves.size;
	for (int move = 0; move < castleSize; move++) {
		movelist.push(makeCastlingMove(castleMoves[move][0][0], castleMoves[move][0][1],
						 			   castleMoves[move][1][0], castleMoves[move][1][1]));
		nodes += movelist.top().perft(depth - 1, movelist);
		movelist.pop();
	}

	Array<std::array<int, 2>, 2>& enPassantMoves = moveset.enPassant;
	int& enPassantSize = enPassantMoves.size;
	for (int move = 0; move < enPassantSize; move++) {
		movelist.push(makeEnPassantMove(enPassantMoves[move][0], enPassantMoves[move][1]));
		nodes += movelist.top().perft(depth - 1, movelist);
		movelist.pop();
	}

	Array<std::array<int, 3>, 8>& doubleMoves = moveset.doubleMoves;
	int& doubleMovesSize = doubleMoves.size;
	for (int move = 0; move < doubleMovesSize; move++) {
		movelist.push(makeDoubleMove(doubleMoves[move][0], doubleMoves[move][1], doubleMoves[move][2]));
		nodes += movelist.top().perft(depth - 1, movelist);
		movelist.pop();
	}

	return nodes;
}

std::string Position::toFen() {
	std::vector<std::string> grid;
	
	for (int i = 63; i >= 0; i--) {
		if ((blackRooks & (1ULL << i)) != 0) {
			grid.push_back("r");
		} else if ((whiteRooks & (1ULL << i)) != 0) {
			grid.push_back("R");
		} else if ((whiteBishops & (1ULL << i)) != 0) {
			grid.push_back("B");
		} else if ((blackBishops & (1ULL << i)) != 0) {
			grid.push_back("b");
		} else if ((blackKnights & (1ULL << i)) != 0) {
			grid.push_back("n");
		} else if ((whiteKnights & (1ULL << i)) != 0) {
			grid.push_back("N");
		} else if ((whiteKing & (1ULL << i)) != 0) {
			grid.push_back("K");
		} else if ((blackKing & (1ULL << i)) != 0) {
			grid.push_back("k");
		} else if ((blackQueens & (1ULL << i)) != 0) {
			grid.push_back("q");
		} else if ((whiteQueens & (1ULL << i)) != 0) {
			grid.push_back("Q");
		} else if ((whitePawns & (1ULL << i)) != 0) {
			grid.push_back("P");
		} else if ((blackPawns & (1ULL << i)) != 0) {
			grid.push_back("p");
		} else {
			grid.push_back(" ");
		}
	}

	int emptyCount = 0;
	int rowCount = 1;
	std::string fen = "";

	for (std::string str : grid) {
		if (str == " ") {
			emptyCount += 1;
		} else {
			if (emptyCount > 0) {
				fen += std::to_string(emptyCount);
				emptyCount = 0;
			}
			fen += str;
			emptyCount = 0;
		}
		if (rowCount % 8 == 0) {
			if (emptyCount > 0) {
				fen += std::to_string(emptyCount);
				emptyCount = 0;
			}
			if (rowCount != 64) {
				fen += "/";
			}
		}
		rowCount += 1;
	}

	fen += isWhiteTurn ? " w" : " b";
	fen += " KQkq ";
	if (enPassant != 0) {
		fen += Tables::chessCoordinates[__builtin_ctzll(enPassant)];
	} else {
		fen += "-";
	}
	fen += " 0 ";
	fen += std::to_string(moveCount);

	return fen;
}