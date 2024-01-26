#include "NNUE.h"
#include "Position.h"
#include <torch/torch.h>
#include "Types.h"

int pvSearch(int alpha, int beta, int depth, std::stack<Position>& movelist, NNUE& model, std::stack<Position>& resMovelist);

inline int searchBlock(bool& bSearchPv, int& beta, int& alpha, int& depth, std::stack<Position>& movelist, NNUE& model, std::stack<Position>& resMovelist) {
	int score = 0;
	if (bSearchPv) {
		score = -pvSearch(-beta, -alpha, depth - 1, movelist, model, resMovelist);
	} else {
		score = -pvSearch(-alpha - 1, -alpha, depth - 1, movelist, model, resMovelist);
		if (score > alpha) {
			score = -pvSearch(-beta, -alpha, depth - 1, movelist, model, resMovelist);
		}
	}
	return score;
}

inline int evaluate(std::stack<Position>& movelist, NNUE& model) {
	std::array<torch::Tensor, 2> halfkp = movelist.top().halfkp();

	torch::Tensor denseHalf1 = halfkp[0].to_dense();
	torch::Tensor denseHalf2 = halfkp[1].to_dense();
	torch::Tensor half1 = denseHalf1.unsqueeze_(0);
	torch::Tensor half2 = denseHalf2.unsqueeze_(0);
	torch::Tensor output = model->forward(half1, half2);
	int eval = output[0][0].item().to<int>();
	return eval;
}

inline int pvSearch(int alpha, int beta, int depth, std::stack<Position>& movelist, NNUE& model, std::stack<Position>& resMovelist) {
	bool bSearchPv = true;
	int score = 0;
    Moveset moveset;
	Position position = movelist.top();

    if (position.isWhiteTurn) {
        position.whiteMoves(moveset);
    } else {
        position.blackMoves(moveset);
    }

	if ((moveset.castle.size == 0 && moveset.doubleMoves.size == 0 && moveset.enPassant.size == 0 && moveset.normal.size == 0 && moveset.promotion.size == 0) 
	|| ((position.whiteKing | position.blackKing) == 0)) {
		return -1000000;
	}

	Position best;
    if (depth == 0) {
        return evaluate(movelist, model);
    }

    Array<std::array<int, 2>, 218>& normalMoves = moveset.normal;
	int& normalSize = normalMoves.size;
	for (int move = 0; move < normalSize; move++) {
		movelist.push(position.makeNormalMove(normalMoves[move][0], normalMoves[move][1]));
		score = searchBlock(bSearchPv, beta, alpha, depth, movelist, model, resMovelist);
		
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			best = movelist.top();
			bSearchPv = false;
		}
		movelist.pop();
	}

	Array<std::array<std::array<int, 2>, 2>, 2>& castleMoves = moveset.castle;
	int& castleSize = castleMoves.size;
	for (int move = 0; move < castleSize; move++) {
		movelist.push(position.makeCastlingMove(castleMoves[move][0][0], castleMoves[move][0][1],
						 		 				castleMoves[move][1][0], castleMoves[move][1][1]));
		score = searchBlock(bSearchPv, beta, alpha, depth, movelist, model, resMovelist);
		
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			best = movelist.top();
			bSearchPv = false;
		}
		movelist.pop();
	}

	Array<std::array<int, 3>, 2>& enPassantMoves = moveset.enPassant;
	int& enPassantSize = enPassantMoves.size;
	for (int move = 0; move < enPassantSize; move++) {
		movelist.push(position.makeEnPassantMove(enPassantMoves[move][0], enPassantMoves[move][1], enPassantMoves[move][2]));
		score = searchBlock(bSearchPv, beta, alpha, depth, movelist, model, resMovelist);
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			best = movelist.top();
			bSearchPv = false;
		}
		movelist.pop();
	}

	Array<std::array<int, 3>, 8>& doubleMoves = moveset.doubleMoves;
	int& doubleMovesSize = doubleMoves.size;
	for (int move = 0; move < doubleMovesSize; move++) {
		movelist.push(position.makeDoubleMove(doubleMoves[move][0], doubleMoves[move][1], doubleMoves[move][2]));
		score = searchBlock(bSearchPv, beta, alpha, depth, movelist, model, resMovelist);
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			best = movelist.top();
			bSearchPv = false;
		}
		movelist.pop();
	}

	Array<std::array<int, 3>, 64>& promotions = moveset.promotion;
	int& promotionSize = promotions.size;
	for (int move = 0; move < promotionSize; move++) {
		movelist.push(position.makePromotionMove(promotions[move][0], promotions[move][1], promotions[move][2]));
		score = searchBlock(bSearchPv, beta, alpha, depth, movelist, model, resMovelist);
		if (score >= beta) {
			return beta;
		}
		if (score > alpha) {
			alpha = score;
			best = movelist.top();
			bSearchPv = false;
		}

		movelist.pop();
	}

	resMovelist.push(best);

	return alpha;
}