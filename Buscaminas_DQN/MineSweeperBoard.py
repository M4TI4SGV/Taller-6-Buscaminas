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
        return self.get_state()

    def get_state(self):
        # Retorna el estado actual del tablero como una matriz de valores:
        # -1 para celdas cerradas, el número de minas cercanas si la celda está abierta, y 9 si es una mina
        state = []
        for i in range(self.width):
            row = []
            for j in range(self.height):
                if self.board.m_Patches[i][j]:  # Celda abierta
                    row.append(self.board.m_Mines[i][j])
                else:
                    row.append(-1)  # Celda cerrada
            state.append(row)
        return np.array(state)

    def step(self, action):
        # Realizar la acción (abrir celda) y devolver el nuevo estado, recompensa, y si el juego terminó
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
        return self.get_state(), reward, self.done
