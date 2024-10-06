import random
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import ReplayBuffer, NUM_ACTIONS, INDEX_TO_ACTION, ACTION_TO_INDEX
from torch.utils.tensorboard import SummaryWriter
from openings import get_opening_moves

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Adição residual
        out = F.relu(out)
        return out


# Inserir aqui a função heuristic_move_selection
def heuristic_move_selection(board):
    legal_moves = list(board.legal_moves)
    # Priorizar movimentos que capturam peças
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    if capture_moves:
        return random.choice(capture_moves)
    # Priorizar movimentos para o centro
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_moves = [move for move in legal_moves if move.to_square in center_squares]
    if center_moves:
        return random.choice(center_moves)
    # Caso contrário, escolher um movimento aleatório
    return random.choice(legal_moves)

class Network(nn.Module):
    def __init__(self, num_actions, num_res_blocks=5):
        super(Network, self).__init__()
        self.num_actions = num_actions
        self.conv = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)

        # Blocos residuais
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_res_blocks)]
        )

        # Cabeça de política
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, self.num_actions)

        # Cabeça de valor
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Entrada: x com dimensão (batch_size, 12, 8, 8)
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)

        # Cabeça de política
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)

        # Cabeça de valor
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class Agent:
    def __init__(self):
        self.num_actions = NUM_ACTIONS  # Adicionado
        self.model = Network(num_actions=self.num_actions)
        self.l2_coeff = 1e-4   # Coeficiente de regularização L2
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=self.l2_coeff)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 0.1  # Taxa de exploração
        self.batch_size = 256  # Defina o tamanho do lote para o treinamento
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.training_step = 0
        self.writer = SummaryWriter(log_dir='../logs')

    def l2_regularization(self):
        l2_loss = 0.0
        for param in self.model.parameters():
            l2_loss += torch.norm(param) ** 2
        return self.l2_coeff * l2_loss


    def select_action_from_mcts(self, root):
        # Seleciona a ação com base nas contagens de visitas da MCTS
        visits = np.array([child.visits for child in root.children.values()])
        actions = list(root.children.keys())
        # Calcula as probabilidades das ações com base nas visitas
        pi = visits / np.sum(visits)
        # Escolhe a ação proporcionalmente às visitas
        action = np.random.choice(actions, p=pi)
        return action, pi

    def select_action(self, board):
        # Verificar se há um movimento de abertura
        opening_move = get_opening_moves(board)
        if opening_move and opening_move in board.legal_moves:
            return opening_move

        # Usar heurística se o agente ainda estiver em fase inicial de treinamento
        if self.training_step < 1000:
            heuristic_move = heuristic_move_selection(board)
            if heuristic_move:
                return heuristic_move

        # Código para seleção de ação usando a rede neural
        state_tensor = self._board_to_tensor(board).unsqueeze(0).to(self.device)
        legal_moves = list(board.legal_moves)
        legal_actions = [ACTION_TO_INDEX[move.uci()] for move in legal_moves if move.uci() in ACTION_TO_INDEX]

        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)

        # Aplicar máscara para movimentos ilegais
        mask = torch.zeros(self.num_actions, dtype=torch.float32)
        mask[legal_actions] = 1

        policy_logits = policy_logits.cpu().squeeze()
        masked_logits = policy_logits + (mask - 1) * 1e10  # Definir logits de movimentos ilegais para um valor muito negativo

        policy = torch.nn.functional.softmax(masked_logits, dim=0).numpy()

        # Selecionar ação com base nas probabilidades
        action_index = np.random.choice(self.num_actions, p=policy)
        action_uci = INDEX_TO_ACTION[action_index]
        return chess.Move.from_uci(action_uci)

    def _random_action(self, legal_moves):
        # Seleciona uma ação aleatória entre os movimentos legais
        return np.random.choice(legal_moves)

    def _best_action(self, board, legal_moves):
        # Avalia todos os movimentos legais e seleciona o melhor com base na rede neural
        best_value = -np.inf
        best_move = None

        for move in legal_moves:
            # Simula o movimento no tabuleiro
            board.push(move)
            state_tensor = self._board_to_tensor(board)
            board.pop()

            # Avalia o estado resultante
            with torch.no_grad():
                value = self.model(state_tensor.to(self.device)).item()

            if value > best_value:
                best_value = value
                best_move = move

        return best_move


    def learn(self, memory):
        print(f"\nIniciando o passo de treinamento {self.training_step + 1}")
        print(f"Tamanho do lote (batch): {len(memory)}")

        # Descompactar o batch
        states, mcts_probs, rewards = zip(*memory)
        states_tensor = torch.stack([self._board_to_tensor(s) for s in states]).to(self.device)
        mcts_probs_tensor = torch.tensor(np.array(mcts_probs), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Zerar os gradientes
        self.optimizer.zero_grad()

        # Passar os estados pelo modelo para obter as previsões
        policy_logits, values = self.model(states_tensor)

        # Calcular a perda de política (Policy Loss)
        policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * F.log_softmax(policy_logits, dim=1), dim=1))
        print(f"Policy Loss: {policy_loss.item():.4f}")

        # Calcular a perda de valor (Value Loss)
        value_loss = F.mse_loss(values.view(-1), rewards_tensor)
        print(f"Value Loss: {value_loss.item():.4f}")

        # Calcular a perda total com regularização
        l2_loss = self.l2_regularization()
        loss = policy_loss + value_loss + l2_loss
        print(f"L2 Regularization Loss: {l2_loss.item():.4f}")
        print(f"Total Loss: {loss.item():.4f}")

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        # Atualizar o contador de passos de treinamento
        self.training_step += 1

        # Registrar a perda no TensorBoard
        self.writer.add_scalar('Loss/Total', loss.item(), self.training_step)
        self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.training_step)
        self.writer.add_scalar('Loss/Value', value_loss.item(), self.training_step)
        self.writer.add_scalar('Loss/L2', l2_loss.item(), self.training_step)

        print(f"Passo de treinamento {self.training_step} concluído.\n")


    def _update_network(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Converter para tensores
        states = torch.stack([self._board_to_tensor(b) for b in states]).to(self.device)
        next_states = torch.stack([self._board_to_tensor(b) for b in next_states]).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Remover dimensões extras se necessário
        actions = actions.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        # Previsão da política e do valor atual
        policy_logits, current_values = self.model(states)
        # Remover dimensões extras
        policy_logits = policy_logits.squeeze()
        current_values = current_values.squeeze()

        # Previsão da política e do valor próximo
        _, next_values = self.model(next_states)
        next_values = next_values.detach().squeeze()

        # Calcular o alvo do valor
        target_values = rewards + self.gamma * next_values * (1 - dones)
        target_values = target_values.squeeze()

        # Verificar as dimensões dos tensores
        #print(f"policy_logits.shape: {policy_logits.shape}")
        #print(f"actions.shape: {actions.shape}")
        #print(f"current_values.shape: {current_values.shape}")
        #print(f"target_values.shape: {target_values.shape}")
        #print(torch.cuda.is_available())


        # Calcular a perda do valor
        value_loss = self.criterion(current_values, target_values)

        # Calcular a perda da política
        policy_loss = torch.nn.functional.cross_entropy(policy_logits, actions)

        # Combinar as perdas
        loss = value_loss + policy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Registrar a perda
        self.writer.add_scalar('Loss', loss.item(), self.training_step)

        # **Salvar o modelo a cada 1000 passos de treinamento**
        if self.training_step % 1000 == 0:
            self.save_model('../models/model.pth')

        self.training_step += 1

        # **Adicionar esta linha para imprimir a perda**
        print(f"Passo de treinamento {self.training_step}, Loss: {loss.item():.4f}")

    def _board_to_tensor(self, board):
            # Converte o estado do tabuleiro em um tensor de dimensão (12, 8, 8)
            state = np.zeros((12, 8, 8), dtype=np.float32)
            piece_map = board.piece_map()
            for square, piece in piece_map.items():
                piece_type = piece.piece_type - 1  # De 0 a 5
                color = int(piece.color)  # 0 para branco, 1 para preto
                index = piece_type + (6 * color)
                row = square // 8
                col = square % 8
                state[index, row, col] = 1


    def _piece_to_value(self, piece):
        # Converter peça para um valor numérico
        piece_values = {
            'P': 1, 'N': 3, 'B': 3.25, 'R': 5, 'Q': 9, 'K': 0,
            'p': -1, 'n': -3, 'b': -3.25, 'r': -5, 'q': -9, 'k': 0
        }
        return piece_values[piece.symbol()]
    
    def save_model(self, path='../models/model.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='../models/model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
