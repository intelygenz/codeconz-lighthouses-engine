<template>
  <div class="scoreboard">
    <transition-group name="player-list" tag="div" class="player-list">
      <div class="player" v-for="player in players" :key="player.id">
        <div class="avatar">
          <img alt="Player avatar" :src="`https://api.dicebear.com/8.x/bottts-neutral/svg?backgroundType=gradientLinear&radius=50&seed=${player.name}`">
        </div>
        <div class="player-data">
          <div class="player-name">{{ player.name }}</div>
          <div class="player-stats">
            <span><i class="fa-solid fa-medal"></i> Score: <strong>{{ player.score }}</strong></span>
            <span><i class="fa-solid fa-bolt energy"></i> Energy: <strong>{{ player.energy }}</strong></span>
          </div>
        </div>
      </div>
    </transition-group>
  </div>
</template>

<script>
export default {
  name: 'PlayerList',
  props: {
    players: Array
  }
}
</script>

<style scoped>
.scoreboard {
  direction: rtl;
  overflow-y: auto;
}

.player-list {
  color: #0A1606;
  direction: ltr;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  padding: 10px;
}

/*.title {*/
/*  color: #A4F20D;*/
/*  text-shadow: 0px 0px 5px rgba(221, 235, 238, .5);*/
/*  text-align: center;*/
/*  font-size: 2.5em;*/
/*  /*font-family: 'JackSphinx', sans-serif;*/
/*  margin: 10px 0;*/
/*}*/

.player {
  color: #F6FAFB;
  display: grid;
  align-items: center;
  grid-template-columns: 50px 1fr;
  height: 50px;
  gap: 10px;
}

.player-data {
  min-width: 0;
}

.avatar img {
  display: block;
}

.player-data {
  height: 100%;
  display: grid;
  grid-template-rows: auto auto;
  gap: 5px;
}

.player-name {
  margin-top: auto;
  font-size: .9em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.player-stats {
  display: flex;
  align-items: flex-start;
  font-size: .75em;
  color: #9EC9EA;
  gap: 10px;
}

.fa-solid {
  font-size: .8em;
}

.fa-medal {
  color: yellow;
}

.fa-bolt {
  color: #10FEFD;
}
</style>
