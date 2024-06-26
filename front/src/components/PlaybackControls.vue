<template>
  <div id="playback-controls">
    <div v-if="playback.currentRound.isStub">
      <p>GameStart</p>
      <p>GameStart</p>
    </div>
    <div v-else>
      <p>Round {{playback.currentRound.id}}</p>
      <p v-if="playback.currentPlayer.isStub">Round start</p>
      <p v-else>Turn {{playback.currentPlayer.name}}</p>
    </div>
    <button @click="prev">Prev</button>
    <button @click="next">Next</button>
    <button @click="play">Play</button>
    <button @click="pause">Pause</button>
    <button @click="restart">Restart</button>
    <!--
      round 1 > turn Alice < round 1
    <button @click="play">Play</button>
    <button @click="pause">Pause</button>
    <button @click="stop">Stop</button>
    -->
  </div>
</template>

<script>
import { Playback } from '@/code/domain.js'

export default {
  name: 'PlayBack',
  props: {
    playback: Playback,
  },
  methods: {
    next() {
      this.playback.next();
    },
    prev() {
      this.playback.prev();
    },
    play() {
      this.playback.play();
    },
    pause() {
      this.playback.pause();
    },
    restart() {
      this.playback.reset();
    }
  }
}
</script>

<style scoped>
#playback-controls {
  color: white;
}
</style>
