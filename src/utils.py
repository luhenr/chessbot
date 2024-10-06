import random
from collections import deque
import numpy as np
import chess

class ReplayBuffer:
    # Implementação existente
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def create_action_mapping():
    actions = set()
    # Gerar todas as combinações possíveis de movimentos
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Incluir todas as promoções possíveis usando constantes inteiras
            for promotion in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                actions.add(move.uci())
    # Incluir movimentos de roque (castling)
    actions.update(['e1g1', 'e1c1', 'e8g8', 'e8c8'])
    action_to_index = {action: idx for idx, action in enumerate(sorted(actions))}
    index_to_action = {idx: action for action, idx in action_to_index.items()}
    return action_to_index, index_to_action

# Criar mapeamentos globais
ACTION_TO_INDEX, INDEX_TO_ACTION = create_action_mapping()
NUM_ACTIONS = len(ACTION_TO_INDEX)
