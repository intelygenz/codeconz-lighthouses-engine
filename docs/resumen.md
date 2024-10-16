# Resumen del Flujo del Programa

### Inicialización del Juego

1. **Recibir información inicial**:
    - Número del jugador.
    - Número total de jugadores.
    - Posición inicial del jugador.
    - Mapa del área jugable.
    - Coordenadas de los faros.

2. **Enviar respuesta**:
    - Nombre del bot.

### Ronda del Juego (Este ciclo se repite para cada ronda)

1. **Incremento de Energía**:
    - Cada casilla incrementa su energía según la fórmula:
      ```energia += floor(5 - distancia_a_faro)```
    - Limitar la energía de cada casilla a un máximo de 100 unidades.

2. **Recolección de Energía**:
    - Cada jugador obtiene la energía de su casilla actual.
    - Si varios jugadores están en la misma casilla, la energía se divide.

3. **Obtención de Claves**:
    - Si un jugador está en un faro y no tiene la clave, la obtiene.

4. **Decremento de Energía de Faros**:
    - La energía de cada faro se decrementa en 10 puntos.
    - Si la energía del faro llega a 0, se vuelve neutro y se eliminan las conexiones.

### Turno del Jugador (Para cada jugador en cada ronda)

1. **Recibir información del estado actual**:
    - Posición del jugador.
    - Puntuación del jugador.
    - Nivel de energía acumulada.
    - Energía disponible en casillas cercanas (radio de 3 unidades).
    - Información de los faros (coordenadas, dueño, nivel de energía, conexiones, clave).

2. **Determinar acción a realizar**:
    - **Pasar el turno**:
      ```json
      { "command": "pass" }
      ```
    - **Moverse a una casilla adyacente**:
      ```json
      { "command": "move", "x": <valor>, "y": <valor> }
      ```
    - **Atacar o recargar un faro**:
      ```json
      { "command": "attack", "energy": <cantidad> }
      ```
    - **Conectar faros**:
      ```json
      { "command": "connect", "destination": [<x>, <y>] }
      ```

3. **Enviar acción seleccionada**.

4. **Recibir resultado de la acción**.

### Fin del Juego

- Detectar condición de EOF en stdin y cerrar correctamente el bot.

## Detalles Adicionales

### Condiciones de Juego

- Los jugadores pueden moverse en ocho direcciones (horizontal, vertical y diagonal).
- Las conexiones entre faros no pueden cruzarse ni pasar por el centro de otro faro.
- Los jugadores no conocen la posición de los demás.

### Puntuación

- Puntos por faros controlados, parejas de faros conectados y triángulos formados.
- Las casillas dentro de triángulos iluminados suman puntos.

## Resumen de Comunicación

### Inicio del Juego

- Motor -> Bot: Información inicial.
- Bot -> Motor: Nombre del bot.

### Durante el Juego

- Motor -> Bot: Estado al comienzo del turno.
- Bot -> Motor: Acción a realizar.
- Motor -> Bot: Resultado de la acción.
- Repetir mientras dure la partida.

### Fin del Juego

- Motor cierra stdin/stdout.