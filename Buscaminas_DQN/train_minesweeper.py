# train_minesweeper.py

import numpy as np
from mine_sweeper_env import MineSweeperEnv  # Importa la clase de entorno
from dqn_agent import DQNAgent  # Importa el agente DQN

if __name__ == "__main__":
    env = MineSweeperEnv(5, 5, 5)  # Tablero de 5x5 con 5 minas
    state_shape = (5, 5, 2)  # Estado incluye tablero y probabilidades
    action_size = 5 * 5  # Número de acciones posibles (una por celda)
    agent = DQNAgent(state_shape, action_size)
    episodes = 1000  # Número de episodios para entrenar

    for e in range(episodes):
        state, probabilities = env.reset()
        combined_state = np.stack((state, probabilities), axis=-1)  # Combina estado y probabilidades
        combined_state = np.reshape(combined_state, (1, *state_shape))
        done = False
        total_reward = 0
        turn = 0

        while not done:
            action = agent.act(combined_state)  # Elegir acción
            (next_state, next_probabilities), reward, done = env.step(action)  # Ejecutar acción
            combined_next_state = np.stack((next_state, next_probabilities), axis=-1)
            combined_next_state = np.reshape(combined_next_state, (1, *state_shape))

            agent.remember(combined_state, action, reward, combined_next_state, done)  # Almacenar experiencia
            combined_state = combined_next_state
            total_reward += reward
            env.save_probabilities_to_file(turn)  # Guardar probabilidades actuales
            turn += 1

            if done:
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
        
        if len(agent.memory) > 32:  # Entrena con memoria suficiente
            agent.replay(32)

    # Guardar el modelo entrenado
    agent.model.save("dqn_minesweeper.h5")
    print("Modelo entrenado y guardado como 'dqn_minesweeper.h5'.")
