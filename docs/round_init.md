Una vez arrancado el juego, se ejecutaran n roundas.
Para cada ronda, se ejecutara un turno por cada jugador registrado.

Energía:
- Siempre en enteros

Al incio de cada ronda:
- Incrementamos energia de las casillas de tierra
  - Aplicamos la fórmula `energía += floor(5 - distancia a faro)` por cada faro con un máx. 100 unidades por casilla
- El jugador obtendrá la energia de la casilla en la que se encuentre
  - Si hay varios jugadores en la misma casilla, la energía se reparte entre ellos, y si hay algún sobrante se pierde
- La casilla en la que se encuentra el jugador pierde la energía y se queda a 0
- Si un jugador está en un faro, obtiene la clave si no la tiene ya
- Todos los fatos pierder 10 puntos de energía
- Si algún faro llega a 0 de energía se convierte en neutro, y desaparecen las conexiones
