import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board.copy()

    def step(self, action):
        self.board.push(action)
        reward = self._evaluate_board()
        done = self.board.is_game_over()
        return self.board.copy(), reward, done

    def _evaluate_board(self):
        # Função de avaliação aprimorada
        material = self._material_score()
        positional = self._positional_score()
        return material + positional

    def _material_score(self):
        # Calcula a pontuação de material
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        white_score = sum(piece_values[piece.piece_type] for piece in self.board.piece_map().values() if piece.color == chess.WHITE)
        black_score = sum(piece_values[piece.piece_type] for piece in self.board.piece_map().values() if piece.color == chess.BLACK)
        return white_score - black_score

    def _positional_score(self):
        # Avalia aspectos posicionais (simplificado)
        # Exemplo: controle do centro
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        white_control = sum(1 for move in self.board.legal_moves if move.to_square in center_squares and self.board.turn == chess.WHITE)
        black_control = sum(1 for move in self.board.legal_moves if move.to_square in center_squares and self.board.turn == chess.BLACK)
        return (white_control - black_control) * 0.1
