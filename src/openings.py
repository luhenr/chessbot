# src/openings.py

import chess

OPENING_MOVES = [
    # Abertura Italiana
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    # Defesa Siciliana
    ["e2e4", "c7c5"],
    # Defesa Francesa
    ["e2e4", "e7e6"],
    # Defesa Caro-Kann
    ["e2e4", "c7c6"],
    # Abertura Ruy Lopez (Espanhola)
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    # Defesa Alekhine
    ["e2e4", "g8f6"],
    # Defesa Moderna
    ["e2e4", "g7g6"],
    # Abertura Inglesa
    ["c2c4"],
    # Abertura Réti
    ["g1f3", "d7d5", "c2c4"],
    # Gambito da Dama
    ["d2d4", "d7d5", "c2c4"],
    # Defesa Índia do Rei
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # Defesa Nimzo-Índia
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # Abertura Escocesa
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    # Defesa Grunfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    # Abertura Bird
    ["f2f4"],
    # Abertura Larsen
    ["b2b3"],
    # Abertura Holandesa
    ["d2d4", "f7f5"],
    # Abertura Trompowsky
    ["d2d4", "g8f6", "c1g5"],
    # Gambito do Rei
    ["e2e4", "e7e5", "f2f4"],
    # Abertura Catalã
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    # Defesa Escandinava
    ["e2e4", "d7d5"],
    # Abertura Vienense
    ["e2e4", "e7e5", "c2c3"],
    # Defesa Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6"],
    # Defesa Benoni Moderna
    ["d2d4", "g8f6", "c2c4", "c7c5"],
    # Defesa Bogo-Índia
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"],
    # Gambito Benko
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4c5", "b7b5"],
    # Gambito Evans
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"],
    # Abertura Colle
    ["d2d4", "d7d5", "e2e3"],
    # Defesa Índia da Dama
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],
    # Abertura do Bispo
    ["e2e4", "e7e5", "f1c4"],
    # Abertura Stonewall
    ["d2d4", "e7e6", "e2e3", "f7f5"],
    # Defesa Eslava
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    # Abertura Inglesa Simétrica
    ["c2c4", "c7c5"],
    # Abertura Grob
    ["g2g4"],
    # Abertura Polaca (Orangotango)
    ["b2b4"],
    # Abertura King's Indian Attack
    ["g1f3", "d7d5", "g2g3"],
    # Defesa Holandesa Leningrado
    ["d2d4", "f7f5", "g2g3", "g7g6"],
    # Gambito Letão
    ["e2e4", "e7e5", "g1f3", "f7f5"],
    # Abertura do Peão da Dama
    ["d2d4", "d7d5"],
    # Abertura Zukertort
    ["g1f3", "d7d5", "b2b3"],
    # Defesa Siciliana Dragon
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g7g6"],
    # Defesa Siciliana Najdorf
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Defesa Siciliana Scheveningen
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e6"],
    # Defesa Petroff
    ["e2e4", "e7e5", "g1f3", "g8f6"],
    # Defesa Philidor
    ["e2e4", "e7e5", "g1f3", "d7d6"],
    # Gambito Smith-Morra
    ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3"],
    # Gambito Escocês
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "f8c5"],
    # Abertura Bird's
    ["f2f4"],
    # Abertura Four Knights
    ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"],
]

def get_opening_moves(board):
    # Verificar se o estado atual do tabuleiro corresponde a alguma abertura
    for opening in OPENING_MOVES:
        if len(board.move_stack) < len(opening):
            # Verificar se os movimentos até agora correspondem à abertura
            match = True
            for i, move in enumerate(board.move_stack):
                if move.uci() != opening[i]:
                    match = False
                    break
            if match:
                # Retornar o próximo movimento da abertura
                next_move_uci = opening[len(board.move_stack)]
                return chess.Move.from_uci(next_move_uci)
    return None
