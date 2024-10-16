Al inicio de cada turno:
- El jugador recibe el estado del turno actual
- El jugador devuelve una de las siguientes acciones
  - No hacer nada
  - Moverse: cambia la posición del jugador
  - Atacar o recargar faro: resta (atacar) o suma (recargar) energia del jugador en el faro
  - Conectar: crea una conexión con otro faro

Conexión de faros:
- Las conexiones no pueden cruzarse
- Las conexiones no puede atravesar otros faros por su centro exacto


```json
{
  "previous_turn_state":{
    "success": false,
    "message": "Player does not have the destination key"
  },
  "position": {
    "x": 1,
    "y": 3
  },
  "score": 36,
  "energy": 66,
  "view": [
    [-1,-1,-1, 0,-1,-1,-1],
    [-1, 0, 0,50,23,50,-1],
    [-1, 0, 0,32,41, 0,-1],
    [ 0, 0, 0, 0,50, 0, 0],
    [-1, 0, 0, 0, 0, 0,-1],
    [-1, 0, 0, 0, 0, 0,-1],
    [-1,-1,-1, 0,-1,-1,-1]
  ],
  "lighthouses": [
    {
      "position": {
        "x": 1,
        "y": 1
      },
      "owner": 0,
      "energy": 30,
      "connections": [[1, 3]],
      "have_key": false
    },
    {
      "position": {
        "x": 1,
        "y": 2
      },
      "owner": -1,
      "energy": 0,
      "connections": [],
      "have_key": false
    },
    {
      "position": {
        "x": 1,
        "y": 3
      },
      "owner": 1,
      "energy": 90,
      "connections": [],
      "have_key": false
    },
    {
      "position": {
        "x": 1,
        "y": 3
      },
      "owner": 0,
      "energy": 50,
      "connections": [[1, 1]],
      "have_key": true
    }
  ]
}
```