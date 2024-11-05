# play_minesweeper.py

import numpy as np
from mine_sweeper_env import MineSweeperEnv  # Importa el entorno
from tensorflow.keras.models import load_model  # Para cargar el modelo guardado
from tensorflow.keras.losses import MeanSquaredError  # Importar la función de pérdida personalizada

# Cargar el modelo entrenado con la función de pérdida personalizada
model = load_model('dqn_minesweeper.h5', custom_objects={'mse': MeanSquaredError()})
print("Modelo entrenado cargado.")

def play_game():
    env = MineSweeperEnv(5, 5, 5)  # Tablero de 5x5 con 5 minas
    state_shape = (5, 5, 2)
    
    state, probabilities = env.reset()
    combined_state = np.stack((state, probabilities), axis=-1)
    combined_state = np.reshape(combined_state, (1, *state_shape))
    done = False
    turn = 0

    while not done:
        while True:
            act_values = model.predict(combined_state)
            flat_action = np.argmax(act_values[0])  # Selecciona la acción con el valor más alto
            action = divmod(flat_action, state_shape[1])  # Convierte a coordenadas (fila, columna)

            # Imprime la acción seleccionada
            print(f"Intentando abrir la celda en {action}")

            # Si la celda está cerrada, realizamos la acción
            if not env.board.m_Patches[action[0]][action[1]]:
                break
            # Si está abierta, fuerza una acción de exploración aleatoria
            print("Celda ya abierta, eligiendo una acción aleatoria")
            action = (np.random.randint(0, env.width), np.random.randint(0, env.height))

        # Ejecutar la acción en el entorno
        (next_state, next_probabilities), reward, done = env.step(action)
        
        # Actualizar el estado combinado
        combined_next_state = np.stack((next_state, next_probabilities), axis=-1)
        combined_next_state = np.reshape(combined_next_state, (1, *state_shape))
        combined_state = combined_next_state

        env.save_probabilities_to_file(turn)  # Guarda las probabilidades actuales en un archivo
        turn += 1

        # Mostrar el tablero y recompensa en la consola
        print(env.board)
        print(f"Turno {turn}, Recompensa: {reward}")

        if done:
            if env.board.have_won():
                print("¡Ganaste!")
            elif env.board.have_lose():
                print("Perdiste :-(")
            break

# Ejecutar el juego
if __name__ == "__main__":
    play_game()
