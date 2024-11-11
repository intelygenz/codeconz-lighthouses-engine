<script setup>

const { playback } = defineProps({
  playback: Playback,
});
</script>

<template>
  <div class="text-white mx-auto">
    <div v-if="playback.isGameStart">
      <p>Game</p>
      <p>Start</p>
    </div>
    <div v-else>
      <p>Round {{ playback.currentFrame.roundIndex + 1 }}</p>
      <div v-if="playback.currentFrame.isRoundFrame">
        <p>Start</p>
      </div>
      <div v-else>
        <p>End of {{ playback.currentFrame.playerName }} turn</p>
      </div>
    </div>

    <!--
        <p v-if="playback.currentPlayer.isStub">Round start</p>
        <p v-else>Turn {{playback.currentTurn.player}}</p>
      <p>GameStart</p>
      <p>GameStart</p>
    </div>
    <div v-else>
      <p>Round {{playback.currentRoundIndex + 1}}</p>
      <p v-if="playback.currentPlayer.isStub">Round start</p>
      <p v-else>Turn {{playback.currentPlayer.name}}</p>
    </div>
    -->
    <div class="flex">
      <button @click="playback.restart">
        <SkipBack />
      </button>
      <button @click="playback.prev">
        <StepBack />
      </button>
      <button v-if="playback.status == PlaybackStatus.done" @click="playback.play">
        <Play />
      </button>
      <button v-else @click="playback.play">
        <Pause />
      </button>
      <button @click="playback.next">
        <StepForward />
      </button>
    </div>
    <!--
      round 1 > turn Alice < round 1
    <button @click="play">Play</button>
    <button @click="pause">Pause</button>
    <button @click="stop">Stop</button>
    -->
  </div>
</template>
