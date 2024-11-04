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
import ScoreBoard from '@/components/ScoreBoard.vue'
import GameBoard from '@/components/GameBoard.vue'
import PlaybackControls from '@/components/PlaybackControls.vue'
import { Playback } from '@/code/domain.js'
import * as games from '@/code/games.js'

const game = games.map_1
const playback = new Playback(game, 5)

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
