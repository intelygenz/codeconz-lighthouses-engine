<template>
  <div class="main">
    <div class="sidebar">
      <div class="header">
        <img alt="Lighthouse logo" src="@/assets/logo.png">
      </div>
      <ScoreBoard :players="orderedPlayers" />
    </div>
    <GameBoard :tiles="tiles" :playback="playback"/>
    <PlaybackControls :playback="plaback"/>
  </div>
</template>

<script>
import ScoreBoard from './components/ScoreBoard.vue'
import GameBoard from './components/GameBoard.vue'
import PlaybackControls from './components/PlaybackControls.vue'

export default {
  name: 'App',
  components: {
    ScoreBoard,
    GameBoard,
    //PlaybackControls
  },
  data() {
    return {
      players: [
        { id: 1, name: 'Alice', points: 10, energy: 100 },
        { id: 2, name: 'Bob', points: 20, energy: 90 },
        { id: 3, name: 'Charlie', points: 30, energy: 80 },
        { id: 4, name: 'David', points: 40, energy: 70 },
        { id: 5, name: 'Eve', points: 50, energy: 60 },
        { id: 6, name: 'Frank', points: 60, energy: 50 },
        { id: 7, name: 'Grace', points: 70, energy: 40 },
        { id: 8, name: 'Heidi', points: 80, energy: 30 },
        { id: 9, name: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore', points: 90, energy: 20 },
        { id: 10, name: 'Judy', points: 100, energy: 10 },
        { id: 11, name: 'Eve 2', points: 50, energy: 60 },
        { id: 12, name: 'Frank 2', points: 60, energy: 50 },
        { id: 13, name: 'Grace 2', points: 70, energy: 40 },
        { id: 14, name: 'Heidi 2', points: 80, energy: 30 },
        { id: 15, name: 'Someone 2', points: 90, energy: 20 },
        { id: 16, name: 'Judy 2', points: 100, energy: 10 },
      ],
      tiles: [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
        [-1,  0, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1],
        [-1,  0, -1, -1,  0,  0,  0,  0,  0, -1, -1,  0, -1],
        [-1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1,  0, -1],
        [-1, -1, -1, -1, -1, -1,  0,  0, -1, -1,  0,  0, -1],
        [-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
        [-1, -1, -1,  0,  0, -1, -1, -1,  0, -1, -1, -1, -1],
        [-1,  0,  0,  0, -1, -1, -1, -1,  0, -1,  0,  0, -1],
        [-1, -1,  0, -1, -1, -1,  0,  0,  0, -1,  0,  0, -1],
        [-1,  0,  0,  0,  0, -1, -1, -1,  0,  0,  0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      ],
    }
  },
  computed: {
    orderedPlayers() {
      return this.players.slice().sort((a, b) => b.points - a.points);
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
</style>
