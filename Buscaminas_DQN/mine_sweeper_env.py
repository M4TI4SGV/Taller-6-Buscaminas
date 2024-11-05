import numpy as np
from MineSweeperBoard import MineSweeperBoard

class MineSweeperEnv:
    def __init__(self, width, height, num_mines):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.reset()

    def reset(self):
        # Reiniciar el tablero y otras variables necesarias para comenzar un nuevo juego
        self.board = MineSweeperBoard(self.width, self.height, self.num_mines)
        self.done = False
        self.probabilities = self.initialize_probabilities()
        return self.get_state()

    def initialize_probabilities(self):
        # Inicializar probabilidades en el tablero con un valor intermedio (0.5) para celdas desconocidas
        return [[0.5 for _ in range(self.height)] for _ in range(self.width)]

    def update_probabilities(self):
        # Actualizar la matriz de probabilidades en función de las celdas abiertas
        for i in range(self.width):
            for j in range(self.height):
                if self.board.m_Patches[i][j] and 0 < self.board.m_Mines[i][j] < 9:
                    num_mines = self.board.m_Mines[i][j]
                    hidden_neighbors = []

                    # Encontrar celdas vecinas que no estén abiertas
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = i + dx, j + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height and not self.board.m_Patches[nx][ny]:
                                hidden_neighbors.append((nx, ny))

                    # Si hay celdas no abiertas, actualizar sus probabilidades
                    if hidden_neighbors:
                        prob = num_mines / len(hidden_neighbors)
                        for (nx, ny) in hidden_neighbors:
                            # Solo actualizar si la celda no ha sido revelada aún
                            self.probabilities[nx][ny] = max(self.probabilities[nx][ny], prob)

    def get_state(self):
        # Retorna el estado actual del tablero y las probabilidades para el agente
        state = []
        for i in range(self.width):
            row = []
            for j in range(self.height):
                if self.board.m_Patches[i][j]:  # Celda abierta
                    row.append(self.board.m_Mines[i][j])
                else:
                    row.append(-1)  # Celda cerrada
            state.append(row)
        return np.array(state), np.array(self.probabilities)

    def step(self, action):
        # Realizar la acción (abrir celda) y devolver el nuevo estado, recompensa y si el juego terminó
        i, j = action
        if self.board.m_Patches[i][j]:  # Acción inválida si ya está abierta
            return self.get_state(), -1, self.done  # Penalización por intentar abrir una celda ya abierta
        
        result = self.board.click(i, j)
        reward = 0
        if self.board.have_lose():
            self.done = True
            reward = -10  # Penalización grande por perder
        elif self.board.have_won():
            self.done = True
            reward = 10  # Recompensa grande por ganar
        else:
            reward = 1 if result == 0 else 0  # Pequeña recompensa por abrir celdas seguras
        self.update_probabilities()  # Actualizar las probabilidades después de cada acción
        return self.get_state(), reward, self.done

    def save_probabilities_to_file(self, turn):
        # Guardar la matriz de probabilidades en un archivo txt
        with open("probabilities_log.txt", "a") as file:
            file.write(f"Turn {turn}:\n")
            for row in self.probabilities:
                file.write(" ".join(f"{cell:.2f}" for cell in row) + "\n")
            file.write("\n")
