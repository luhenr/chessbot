# src/gui.py

import tkinter as tk
from PIL import Image, ImageTk
import chess
import chess.svg
import io
from agent import Agent
from chess_env import ChessEnv
import cairosvg  # Importar o cairosvg para conversão de SVG para PNG
import os

class ChessGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chess Agent Self-Play")
        self.env = ChessEnv()

        # Inicializar os agentes
        self.agent = Agent()
        self.opponent = Agent()
        self.agent.epsilon = 0.0  # Desativar exploração na GUI
        self.opponent.epsilon = 0.0

        # Caminho para o modelo
        self.model_path = '../models/model.pth'

        # Carregar o modelo
        self.load_model()

        # Iniciar recarregamento periódico do modelo
        self.root.after(5000, self.reload_model)  # Recarregar a cada 5 segundos

        self.canvas = tk.Canvas(self.root, width=480, height=480)
        self.canvas.pack()
        self.update_board()

        self.root.after(1000, self.play_game)
        self.root.mainloop()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.agent.load_model(self.model_path)
                self.opponent.load_model(self.model_path)
                print("Modelo carregado com sucesso.")
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
        else:
            print(f"Arquivo de modelo {self.model_path} não encontrado.")

    def reload_model(self):
        self.load_model()
        self.root.after(5000, self.reload_model)  # Agendar o próximo recarregamento

    def update_board(self):
        # Renderizar o tabuleiro atual em SVG
        svg_board = chess.svg.board(board=self.env.board).encode('utf-8')

        # Converter SVG para PNG usando cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_board)

        # Abrir a imagem PNG com PIL
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((480, 480), resample=Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.root.update_idletasks()

    def play_game(self):
        if not self.env.board.is_game_over():
            if self.env.board.turn == chess.WHITE:
                action = self.agent.select_action(self.env.board)
            else:
                action = self.opponent.select_action(self.env.board)

            self.env.step(action)
            self.update_board()

            # Aguardar um curto período para visualizar o movimento
            self.root.after(500, self.play_game)
        else:
            print("Jogo terminado!")
            print("Resultado:", self.env.board.result())
            # Reiniciar o jogo após alguns segundos
            self.root.after(2000, self.reset_game)

    def reset_game(self):
        self.env.reset()
        self.update_board()
        self.root.after(1000, self.play_game)

if __name__ == "__main__":
    gui = ChessGUI()
