Al inicio de la partido devolvemos el estado inicial del juego a cada jugador.

- Bloqueamos la petición de Join hasta que el timeout salte
- Calculamos el estado inicial (cual es el timeout?)
- Devolvemos a los players registrados.

Representación del estado inicial
```json
{
  "player_num": 2,
  "player_count": 10,
  "position": {
    "x": 0,
    "y": 0
  },
  "map":  [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]],
  "lighthouses": [
    {
      "position": {
        "x": 100,
        "y": 100
      }
    },
    {
      "position": {
        "x": 200,
        "y": 200
      }
    }
  ]
}
```