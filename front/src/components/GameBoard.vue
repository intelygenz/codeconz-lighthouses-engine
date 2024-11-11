<template>
  <div ref="container" class="w-full grow"></div>
</template>

<script>
import * as pixi from "pixi.js";
import { Board } from "@/code/presentation";
import { Game, Playback, PlaybackStatus } from "@/code/domain";

export default {
  name: "GameBoard",
  props: {
    game: Game,
    playback: Playback,
  },
};
</script>

<style scoped>
.game-board {
  width: 100%;
  height: 100%;
  display: flex;
}
</style>
