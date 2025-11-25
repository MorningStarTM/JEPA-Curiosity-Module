import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []

    def remember(self, state, actions, next_state):
        self.actions.append(actions)
        self.states.append(state)
        self.next_states.append(next_state)

    def clear_memory(self):
        self.states = []
        self.next_states = []
        self.actions = []

    def sample_memory(self):
        return self.states, self.actions, self.next_states, 
    