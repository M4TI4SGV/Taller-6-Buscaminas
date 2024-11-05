import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Factor de descuento para la recompensa futura
        self.epsilon = 1.0  # Factor de exploración inicial
        self.epsilon_min = 0.05  # Reduce menos la exploración
        self.epsilon_decay = 0.99  # Disminuye la tasa de decaimiento de exploración
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Crear red neuronal para aproximar la función Q
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Almacenar experiencia en la memoria
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Tomar una acción basada en exploración o explotación
        if np.random.rand() <= self.epsilon:
            # Exploración: elegir una acción aleatoria
            return (random.randint(0, self.state_shape[0] - 1), random.randint(0, self.state_shape[1] - 1))
        
        # Explotación: elegir la mejor acción según el modelo
        act_values = self.model.predict(state)
        flat_action = np.argmax(act_values[0])  # Acción en formato plano
        return divmod(flat_action, self.state_shape[1])  # Convertir a formato (fila, columna)

    def replay(self, batch_size):
        # Entrenar la red con muestras de la memoria
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            flat_action = action[0] * self.state_shape[1] + action[1]  # Acción en formato plano
            target_f[0][flat_action] = target  # Actualizar el valor objetivo para la acción específica
            
            # Ajustar el modelo a los valores objetivo
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Reducir la tasa de exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

