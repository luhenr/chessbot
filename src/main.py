# src/main.py

from torch.utils.tensorboard import SummaryWriter
import torch
from agent import Agent
from chess_env import ChessEnv
import chess
from mcts import mcts_search, MCTSNode
import numpy as np

def get_game_result(board):
    # Retorna +1 para vitória das brancas, -1 para vitória das pretas, 0 para empate
    result = board.result()
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0

def main():
    env = ChessEnv()
    agent = Agent()
    writer = SummaryWriter(log_dir='logs')

    num_episodes = 1000
    memory = []

    for episode in range(num_episodes):
        print(f"Iniciando o episódio {episode}")
        env.reset()
        done = False
        game_history = []
        move_count = 0

        while not done:
            move_count += 1
            #print(f"Movimento {move_count}")
            # Executar MCTS para obter a ação e as probabilidades
            #print("Executando MCTS...")
            root = MCTSNode(env.board)
            mcts_search(root, agent.model, simulations=10, c_puct=1.0, device=agent.device)
            #print("MCTS concluído.")
            action, pi = agent.select_action_from_mcts(root)
            next_state, reward, done = env.step(action)
            game_history.append((env.board.copy(), pi))

        # Determinar o resultado da partida
        result = get_game_result(env.board)
        print(f"Resultado da partida: {result}")
        # Armazenar os dados para treinamento
        for state, pi in game_history:
            memory.append((state, pi, result))

        print(f"Tamanho da memória: {len(memory)}")

        # Treinar o modelo após acumular um certo número de partidas
        if len(memory) >= agent.batch_size:
            print("Chamando o método learn...")
            agent.learn(memory)
            memory = []  # Limpar a memória após o treinamento

        # Registro no TensorBoard
        writer.add_scalar('Resultado da Partida', result, episode)
        if episode % 10 == 0:
            print(f"Episódio {episode}, Resultado: {result}")

    # Salvar o modelo treinado
    agent.save_model('../models/model.pth')
    writer.close()


if __name__ == "__main__":
    main()
