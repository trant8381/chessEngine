#include "Position.h"
#include "Types.h"

int pvSearch(int alpha, int beta, int depth, std::stack<Position>& movelist);

int searchBlock(bool bSearchPv, int beta, int alpha, int depth, std::stack<Position> movelist) {
	int score = 0;
	if (bSearchPv) {
		score = -pvSearch(-beta, -alpha, depth - 1, movelist);
	} else {
		score = -pvSearch(-alpha - 1, -alpha, depth - 1, movelist);
		if ( score > alpha ) {
			score = -pvSearch(-beta, -alpha, depth - 1, movelist);
		}
	}

	return alpha;
}

int evaluate(int alpha, int beta) {
	return 1;
}

int pvSearch(int alpha, int beta, int depth, std::stack<Position>& movelist) {
    if (depth == 0) {
        return evaluate(alpha, beta);
    }

    bool bSearchPv = true;
	int score = 0;
    Moveset moveset;
	Position position = movelist.top();
    if (position.isWhiteTurn) {
        position.whiteMoves(moveset);
    } else {
        position.blackMoves(moveset);
    }

    Array<std::array<int, 2>, 218>& normalMoves = moveset.normal;
	int& normalSize = normalMoves.size;
	for (int move = 0; move < normalSize; move++) {
		movelist.push(position.makeNormalMove(normalMoves[move][0], normalMoves[move][1]));
		searchBlock(bSearchPv, beta, alpha, depth, movelist);
		movelist.pop();
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			bSearchPv = false;
		}
	}

	Array<std::array<std::array<int, 2>, 2>, 2>& castleMoves = moveset.castle;
	int& castleSize = castleMoves.size;
	for (int move = 0; move < castleSize; move++) {
		movelist.push(position.makeCastlingMove(castleMoves[move][0][0], castleMoves[move][0][1],
						 		 				castleMoves[move][1][0], castleMoves[move][1][1]));
		searchBlock(bSearchPv, beta, alpha, depth, movelist);
		movelist.pop();
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			bSearchPv = false;
		}
	}

	Array<std::array<int, 3>, 2>& enPassantMoves = moveset.enPassant;
	int& enPassantSize = enPassantMoves.size;
	for (int move = 0; move < enPassantSize; move++) {
		movelist.push(position.makeEnPassantMove(enPassantMoves[move][0], enPassantMoves[move][1], enPassantMoves[move][2]));
		searchBlock(bSearchPv, beta, alpha, depth, movelist);
		movelist.pop();
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			bSearchPv = false;
		}
	}

	Array<std::array<int, 3>, 8>& doubleMoves = moveset.doubleMoves;
	int& doubleMovesSize = doubleMoves.size;
	for (int move = 0; move < doubleMovesSize; move++) {
		movelist.push(position.makeDoubleMove(doubleMoves[move][0], doubleMoves[move][1], doubleMoves[move][2]));
		searchBlock(bSearchPv, beta, alpha, depth, movelist);
		movelist.pop();
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			bSearchPv = false;
		}
	}

	Array<std::array<int, 3>, 64>& promotions = moveset.promotion;
	int& promotionSize = promotions.size;
	for (int move = 0; move < promotionSize; move++) {
		movelist.push(position.makePromotionMove(promotions[move][0], promotions[move][1], promotions[move][2]));
		searchBlock(bSearchPv, beta, alpha, depth, movelist);
		movelist.pop();
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			bSearchPv = false;
		}
	}

	return alpha;
}