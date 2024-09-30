<template>
  <div class="main">
    <div class="sidebar">
      <div class="header">
        <img alt="Lighthouse logo" src="@/assets/logo.png">
      </div>
      <ScoreBoard :players="orderedPlayers" />
    </div>
    <div class="game">
      <GameBoard :game="game" :playback="playback"/>
      <PlaybackControls :playback="playback"/>
    </div>
  </div>
</template>

<script>
import ScoreBoard from './components/ScoreBoard.vue'
import GameBoard from './components/GameBoard.vue'
import PlaybackControls from './components/PlaybackControls.vue'

import { 
  Game, 
  Playback,
  player, 
  lighthouse, 
  round,
  initialRoundStatus,
  boardStatus,
  playerStatus,
  playerScore,
  lighthouseStatus,
  turn,
  c
} from './code/domain.js'

const palette = [
  0xff0000, 
  0x580aff,
  0xdeff0a,
  0xbe0aff,
  0x0aefff,
  0xa1ff0a,
  0xff8700,
  0x147df5,
  0x0aff99,
  0xffd300,
]

const matrix = [
  ['w', 'w', 'w', 'w', 'w'],
  ['w', 'g', 'g', 'g', 'w'],
  ['w', 'g', 'g', 'g', 'w'],
  ['w', 'g', 'g', 'g', 'w'],
  ['w', 'w', 'w', 'w', 'w'],
]

const player1 = player(1, 'Alice', palette[0], c(1, 1))
const player2 = player(2, 'Bob', palette[1], c(1, 2))
const player3 = player(3, 'X', palette[2], c(1, 3))
let players = [player1, player2]
players = [player1, player2, player3]

const lighthouse1 = lighthouse(1, c(2, 2))
const lighthouses = [lighthouse1]

let rounds = [
  round(
    1, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 0, 4, 4, 0],
        [0, 0, 5, 100, 0],
        [0, 4, 4, 50, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 4, []), playerStatus(player2.id, 4, [])],
      [playerScore(player1.id, 0), playerScore(player2.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 4, []), c(2, 1)),
      turn(playerStatus(player2.id, 4, []), c(1, 3)),
    ],
  ),
  round(
    2, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 4, 0, 8, 0],
        [0, 4, 10, 100, 0],
        [0, 0, 8, 50, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 12, []), playerStatus(player2.id, 12, [])],
      [playerScore(player1.id, 0), playerScore(player2.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 12, []), c(2, 2)),
      turn(playerStatus(player2.id, 12, []), c(2, 3)),
    ],
  ),
  round(
    3, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 8, 4, 12, 0],
        [0, 8, 0, 100, 0],
        [0, 4, 0, 50, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 27, [1]), playerStatus(player2.id, 24, [])],
      [playerScore(player1.id, 2), playerScore(player2.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 0, [1]), c(2, 2), lighthouseStatus(lighthouse1.id, 1, 27, [])),
      turn(playerStatus(player2.id, 24, []), c(2, 2)),
    ],
  ),
  round(
    4, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 12, 8, 16, 0],
        [0, 12, 0, 100, 0],
        [0, 8, 4, 50, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 2, [1]), playerStatus(player2.id, 26, [1])],
      [playerScore(player1.id, 2), playerScore(player2.id, 2)],
      [lighthouseStatus(lighthouse1.id, 17)]
    ), [
      turn(playerStatus(player1.id, 2, [1]), c(1, 2)),
      turn(playerStatus(player2.id, 0, [1]), c(2, 2), lighthouseStatus(lighthouse1.id, 2, 9, [])),
    ],
  ),
]

rounds = [
  round(
    1, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
      [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 4, []), c(2, 2)),
      turn(playerStatus(player2.id, 4, []), c(2, 2)),
      turn(playerStatus(player3.id, 4, []), c(2, 2)),
    ],
  ),
  round(
    2, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
      [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 12, []), c(3, 3)),
      turn(playerStatus(player2.id, 12, []), c(3, 2)),
      turn(playerStatus(player3.id, 12, []), c(3, 1)),
    ],
  ),
  round(
    3, 
    initialRoundStatus(
      boardStatus([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
      ]),
      [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
      [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
      [lighthouseStatus(lighthouse1.id, 0)]
    ), [
      turn(playerStatus(player1.id, 12, []), c(3, 2)),
      turn(playerStatus(player2.id, 12, []), c(3, 2)),
      turn(playerStatus(player3.id, 12, []), c(3, 2)),
    ],
  )
]

const game = new Game(matrix, players, lighthouses, rounds)
const playback = new Playback(game, 50)

export default {
  name: 'App',
  components: {
    ScoreBoard,
    GameBoard,
    PlaybackControls
  },
  data() {
    return { game, playback }
  },
  computed: {
    orderedPlayers() {
      return this.game.players.slice().sort((a, b) => b.score - a.score);
    },
  },
}
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
}

#app {
  background-color: #0A1606;
  font-family: 'Space Grotest', 'Noto Sans', sans-serif;
}

.main {
  display: flex;
  height: 100vh; /* 100% viewport height */
  width: 100vw; /* 100% viewport width */
}

.sidebar {
  flex: 0 0 400px; /* fixed width */
  display: flex;
  flex-direction: column;
  border-right: 2px solid #12230d;
}

.header {
  flex: 0 0 120px;
  display: flex;
  justify-content: center;
  align-items: center;
  border-bottom: 2px solid #12230d;
  /*background: linear-gradient(to top, white, black 5px, black 6px, #0A1606 6px, #0A1606);*/
}

.header img {
  height: 100px;
}

.game {
  flex: 1; /* take up remaining space */
  display: flex;
  flex-direction: column;
  background-color: black;
}
</style>
