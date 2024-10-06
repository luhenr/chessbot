# src/mcts.py

import chess
import numpy as np
import torch
from utils import ACTION_TO_INDEX

def board_to_tensor(board):
    # Função para converter o tabuleiro em tensor
    state = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1  # De 0 a 5
        color = int(piece.color)  # 0 para branco, 1 para preto
        index = piece_type + (6 * color)
        row = square // 8
        col = square % 8
        state[index, row, col] = 1
    return torch.tensor(state, dtype=torch.float32)

def action_to_index(action):
    # Função para converter uma ação em um índice
    return ACTION_TO_INDEX[action.uci()]

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = prior  # Probabilidade inicial do modelo

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_priors):
        for action, prior in action_priors:
            if action not in self.children:
                next_board = self.board.copy()
                next_board.push(action)
                self.children[action] = MCTSNode(next_board, parent=self, prior=prior)

    def select(self, c_puct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            u = c_puct * child.prior * np.sqrt(self.visits) / (1 + child.visits)
            score = child.value / (1 + child.visits) + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def update(self, value):
        self.visits += 1
        self.value += value

def mcts_search(root, model, simulations, c_puct, device):
    for sim in range(simulations):
        node = root
        search_path = [node]

        # Fase de seleção
        while not node.is_leaf():
            action, node = node.select(c_puct)
            search_path.append(node)

        # Fase de expansão e avaliação
        state_tensor = board_to_tensor(node.board).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().detach().numpy()[0]
        legal_moves = list(node.board.legal_moves)
        action_priors = []
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in ACTION_TO_INDEX:
                idx = ACTION_TO_INDEX[move_uci]
                prior = policy[idx]
                action_priors.append((move, prior))
        node.expand(action_priors)

        # Valor do nó terminal
        leaf_value = value.item()

        # Fase de retropropagação
        for node in reversed(search_path):
            node.update(leaf_value)
            leaf_value = -leaf_value  # Inverte o valor para o oponente

        """if sim % 10 == 0:
            print(f"Simulação MCTS {sim}/{simulations} concluída.")"""

    return root

