#!/usr/bin/python
# -*- coding: utf-8 -*-

# ==============================================================================
# ROBOT
# Los robots definidos deben heredar de esta clase.
# ==============================================================================

class Bot(object):
    """Bot base. Este bot no hace nada (pasa todos los turnos)."""
    NAME = "NullBot"

    # ==========================================================================
    # Comportamiento del bot
    # Métodos a implementar / sobreescribir (opcionalmente)
    # ==========================================================================

    def __init__(self):
        """Inicializar el bot: llamado al comienzo del juego."""
        self.transitions = []
        self.scores = []
        self.player_num = 0
        self.game_map = None

    def initialize_game(self, state):
        pass

    def play(self, state):
        """Jugar: llamado cada turno.
        Debe devolver una acción (jugada).
        
        state: estado actual del juego.
        """
        return self.nop()

    def success(self):
        """Éxito: llamado cuando la jugada previa es válida."""
        pass

    def error(self, message, last_move):
        """Error: llamado cuando la jugada previa no es válida."""
        print("Recibido error: %s", message)
        print("Jugada previa: %r", last_move)

    # ==========================================================================
    # Jugadas posibles
    # No es necesario sobreescribir estos métodos.
    # ==========================================================================

    def nop(self):
        """Pasar el turno"""
        return {
            "command": "pass",
        }

    def move(self, x, y):
        """Mover a una casilla adyacente
        x: delta x (0, -1, 1)
        y: delta y (0, -1, 1)
        """
        return {
            "command": "move",
            "x": x,
            "y": y
        }

    def attack(self, energy):
        """Atacar a un faro
        energy: energía (entero positivo)
        """
        return {
            "command": "attack",
            "energy": energy
        }

    def connect(self, destination):
        """Conectar a un faro remoto
        destination: tupla o lista (x,y): coordenadas del faro remoto
        """
        return {
            "command": "connect",
            "destination": destination
        }