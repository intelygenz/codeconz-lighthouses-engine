<script setup>
  import { reactive, computed, onMounted, useTemplateRef } from "vue";
  import * as Icon from "lucide-vue-next";
  import { Playback, PlaybackStatus } from "@/code/domain";
  import { colorToHex, colorToString } from "@/code/palette";
  import { init } from "@/code/presentation";
  import * as Games from "@/code/maps";
  // import * as EngineGames from "@/code/engine_games";

  const game = reactive(Games.game_1);
  const playback = reactive(new Playback(game, 5));
  const boardContainer = useTemplateRef("boardContainer");

  onMounted(async () => {
    const { board, ticker } = await init(game, playback, boardContainer.value);
    playback.init(board, ticker);
  });
</script>

<template>
  <!-- Main -->
  <div class="flex h-screen w-screen bg-black">
    <!-- Left column -->
    <div class="flex w-80 flex-col bg-[#0a1606]">
      <!-- Header -->
      <div class="flex justify-center border-b-2 border-[#12230d] py-4">
        <img
          alt="Lighthouse logo"
          src="@/assets/logo-sin-fondo.png"
          class="w-28" />
      </div>
      <!-- Header -->

      <!-- Scoreboard -->
      <transition-group
        name="player-list"
        tag="div"
        class="flex w-full flex-col gap-4 p-4 text-slate-100">
        <div
          class="flex gap-4"
          v-for="player in game.orderedPlayers"
          :key="player.id">
          <img
            class="w-12"
            alt="Player avatar"
            :src="`https://api.dicebear.com/8.x/bottts-neutral/svg?backgroundColor=${colorToString(player.color)}&radius=50&seed=${player.name}`" />
          <div class="flex grow flex-col justify-center">
            <span class="text-sm font-bold">{{ player.name }}</span>
            <div class="flex gap-3 text-xs">
              <span class="flex items-center gap-1 text-yellow-300">
                <Icon.Medal class="w-4" />
                <span>{{ player.score }}</span>
              </span>
              <span class="flex items-center gap-1 text-blue-300">
                <Icon.Zap class="w-4" />
                <span>{{ player.energy }}</span>
              </span>
            </div>
          </div>
        </div>
      </transition-group>
      <!-- Scoreboard -->
    </div>
    <!-- Left column -->

    <!-- Center column -->
    <div class="flex grow flex-col bg-black py-4 text-slate-100">
      <!-- Game info -->
      <div class="flex basis-1/12 flex-col justify-end text-center">
        <p class="font-bold">{{ playback.currentFrame.title }}</p>
        <p>{{ playback.currentFrame.subtitle }}</p>
      </div>

      <!-- Board -->
      <div ref="boardContainer" class="w-full grow"></div>
      <!-- Board -->

      <!-- Playback controls -->
      <div class="mx-auto basis-1/12">
        <div class="flex gap-2">
          <button @click="playback.restart">
            <Icon.SkipBack />
          </button>
          <button @click="playback.prev">
            <Icon.StepBack />
          </button>
          <button v-if="playback.isPlaying" @click="playback.stop">
            <Icon.Pause />
          </button>
          <button v-else @click="playback.play">
            <Icon.Play />
          </button>
          <button @click="playback.next">
            <Icon.StepForward />
          </button>
        </div>
      </div>
      <!-- Playback controls -->
    </div>
  </div>
</template>

<style scoped>
  .player-list-move {
    transition: all 0.25s ease;
  }

  .scoreboard {
    direction: rtl;
    overflow-y: auto;
  }

  .player-list {
    color: #0a1606;
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
    display: grid;
    align-items: center;
    grid-template-columns: 50px 1fr;
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
    padding: 2px;
    margin-top: auto;
    font-size: 0.9em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .player-stats {
    display: flex;
    align-items: flex-start;
    font-size: 0.75em;
    color: #9ec9ea;
    gap: 10px;
  }

  .fa-solid {
    font-size: 0.8em;
  }

  .fa-medal {
    color: yellow;
  }

  .fa-bolt {
    color: #10fefd;
  }
</style>
