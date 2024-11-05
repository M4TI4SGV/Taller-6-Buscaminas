import sys
import random
from MineSweeperBoard import *

# Función para crear una matriz inicial de probabilidades
def initialize_probabilities(board):
    width, height = board.width(), board.height()
    return [[0.5 for _ in range(height)] for _ in range(width)]  # Inicializar con una probabilidad intermedia

# Función para actualizar la matriz de probabilidades
def update_probabilities(board, probabilities):
    # Ajustar las probabilidades basadas en las celdas abiertas
    for i in range(board.width()):
        for j in range(board.height()):
            if board.m_Patches[i][j] and 0 < board.m_Mines[i][j] < 9:
                num_mines = board.m_Mines[i][j]
                hidden_neighbors = []

                # Encontrar celdas vecinas que no estén abiertas
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < board.width() and 0 <= ny < board.height() and not board.m_Patches[nx][ny]:
                            hidden_neighbors.append((nx, ny))

                # Si hay celdas no abiertas, actualizar sus probabilidades
                if hidden_neighbors:
                    prob = num_mines / len(hidden_neighbors)
                    for (nx, ny) in hidden_neighbors:
                        # Solo actualizar si la celda no ha sido revelada aún
                        probabilities[nx][ny] = max(probabilities[nx][ny], prob)

# Función para encontrar la celda con menor probabilidad
def find_lowest_probability_cell(board, probabilities):
    min_prob = float('inf')
    best_cell = None
    for i in range(board.width()):
        for j in range(board.height()):
            if not board.m_Patches[i][j] and probabilities[i][j] < min_prob:
                min_prob = probabilities[i][j]
                best_cell = (i, j)
    return best_cell

# Función para guardar la matriz de probabilidades en un archivo txt
def save_probabilities_to_file(probabilities, turn):
    with open("probabilities_log.txt", "a") as file:
        file.write(f"Turn {turn}:\n")
        for row in probabilities:
            file.write(" ".join(f"{cell:.2f}" for cell in row) + "\n")
        file.write("\n")

# Iniciar el juego
if len(sys.argv) < 4:
    print("Usage: python3", sys.argv[0], "width height mines")
    sys.exit(1)

w = int(sys.argv[1])
h = int(sys.argv[2])
m = int(sys.argv[3])
board = MineSweeperBoard(w, h, m)
probabilities = initialize_probabilities(board)

turn = 0
while not board.have_won() and not board.have_lose():
    print(board)
    update_probabilities(board, probabilities)
    save_probabilities_to_file(probabilities, turn)
    turn += 1
    i, j = find_lowest_probability_cell(board, probabilities)
    if i is not None and j is not None:
        board.click(i, j)

print(board)
if board.have_won():
    print("You won!")
elif board.have_lose():
    print("You lose :-(")
