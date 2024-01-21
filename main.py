import chess
import chess.pgn
import chess.engine
from sys import stdout
pgn = open("./data/games.pgn")
engine = chess.engine.SimpleEngine.popen_uci("./stockfish")
games = 1000

with open("./data/positions.txt", 'a+') as moves:
    with open("./data/evals.txt", 'a+') as evals:
        gameNum = 0
        while game := chess.pgn.read_game(pgn):
            board = game.board()

            for move in game.mainline_moves():
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(time=0.001))
                evals.write(str(info["score"].relative) + " ")
                moves.write(board.fen() + "\n")

            evals.write("\n")
            moves.write("\n")

            if gameNum == games:
                break
            gameNum += 1
            stdout.write(str(gameNum) + "\n")
engine.quit()
