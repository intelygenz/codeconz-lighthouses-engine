<script setup lang="ts">
  import engineGame from "./games/game-2024_11_13_18_09_00.json";
  import { computed, onMounted, reactive, useTemplateRef } from "vue";
  import * as Icon from "lucide-vue-next";
  import { Stage, HoverTile } from "./code/presentation";
  import { EngineGame, mapGame } from "./code/engine_mapper";
  import { Playback, MaxSpeed } from "./code/playback";
  import { TileType } from "./code/domain";
  import { colorToString } from "./code/palette";
  import Slider from "@vueform/slider";

  var tailwind = [
    "text-[#ff0000]",
    "text-[#ff8700]",
    "text-[#ffd300]",
    "text-[#deff0a]",
    "text-[#a1ff0a]",
    "text-[#0aff99]",
    "text-[#0aefff]",
    "text-[#147df5]",
    "text-[#580aff]",
    "text-[#be0aff]",
    "text-[#cccccc]",
  ];

  var bg = [
    "bg-[#ff0000]",
    "bg-[#ff8700]",
    "bg-[#ffd300]",
    "bg-[#deff0a]",
    "bg-[#a1ff0a]",
    "bg-[#0aff99]",
    "bg-[#0aefff]",
    "bg-[#147df5]",
    "bg-[#580aff]",
    "bg-[#be0aff]",
    "bg-[#cccccc]",
  ];

  const game = mapGame(engineGame as EngineGame);
  const hoverTile = reactive<HoverTile>({ x: undefined, y: undefined });
  const stage = new Stage(game, hoverTile);

  const playback = new Playback(game, MaxSpeed);
  const status = playback.status;
  const hover = computed(() => status.value.hoverInfo(hoverTile));

  const stageContainer = useTemplateRef("stageContainer");
  onMounted(async () => {
    if (!stageContainer.value) return;
    stage.init(stageContainer.value, playback);
  });
</script>

<template>
  <!-- Loader -->
  <!--
    <div v-show="loading" class="flex h-screen items-center justify-center">
      <Icon.Loader class="w-12 text-slate-100" />
    </div>
    <!-- Loader -->

  <!-- Main -->
  <div class="flex h-screen w-screen bg-black text-slate-100">
    <!-- Left column -->
    <div class="flex w-80 flex-col bg-[#0a1606]">
      <!-- Header -->
      <div
        class="flex h-40 items-center justify-center border-b-2 border-[#12230d]">
        <img
          alt="Lighthouse logo"
          src="@/assets/logo-sin-fondo.png"
          class="h-28 w-28" />
      </div>
      <!-- Header -->

      <!-- Scoreboard -->
      <transition-group
        name="list"
        tag="div"
        class="flex flex-col gap-4 p-4 text-slate-100">
        <div
          class="flex gap-4"
          v-for="player in status.scoreboard"
          :key="player.id">
          <img
            class="w-12"
            alt="Player avatar"
            :src="`https://api.dicebear.com/8.x/bottts-neutral/svg?backgroundColor=${colorToString(player.color)}&radius=25&seed=${player.id}`" />
          <div class="flex grow flex-col justify-center">
            <span
              class="text-sm font-bold"
              :class="[tailwind[player.id - 1]]"
              >{{ player.name }}</span
            >
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

            <div
              class="flex items-center gap-1 text-wrap text-xs text-slate-400">
              <Icon.Key class="w-4" />
              <span v-if="player.keys.length == 0">No keys</span>
              <span v-else v-for="key in player.keys">{{ key }}</span>
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
        <p class="font-bold">{{ status.title }}</p>
        <p>{{ status.subtitle }}</p>
      </div>

      <!-- Board -->
      <div ref="stageContainer" class="w-full grow"></div>
      <!-- Board -->

      <!-- Playback controls -->
      <div class="basis-1/5 space-y-2 text-center">
        <div class="flex justify-center gap-2">
          <button @click="playback.restart">
            <Icon.SkipBack />
          </button>
          <button @click="playback.prev">
            <Icon.StepBack />
          </button>
          <button v-if="status.started" @click="playback.pause">
            <Icon.Pause />
          </button>
          <button v-else @click="playback.play">
            <Icon.Play />
          </button>
          <button @click="playback.next">
            <Icon.StepForward />
          </button>
        </div>

        <div
          class="mx-auto flex items-center justify-center gap-4 text-slate-400">
          <span>Slower</span>
          <Slider
            v-model="playback.speed"
            :min="1"
            :max="MaxSpeed"
            :lazy="false"
            :tooltips="false"
            class="w-40" />
          <span>Faster</span>
        </div>

        <div
          class="mx-auto flex items-center justify-center gap-4 text-slate-400">
          <span>Frame</span>
          <Slider
            v-model="playback.frame"
            :min="0"
            :max="playback.frames.length - 1"
            :lazy="false"
            :tooltips="false"
            :disabled="status.started"
            class="w-80" />
        </div>

        <div v-if="hover.show" class="text-xs">
          <div v-if="hover.tile.type == TileType.Ground">
            <div class="flex items-center justify-center gap-2">
              <span>Tile({{ hover.tile?.x }},{{ hover.tile?.y }})</span>
              <div class="flex items-center gap-1 text-blue-300">
                <Icon.Zap class="w-3" />
                <span>{{ hover.tile?.energy }}</span>
              </div>
            </div>
            <div
              v-if="hover.lighthouse"
              class="flex items-center justify-center gap-2">
              <span :class="[tailwind[hover.lighthouse.ownerId - 1]]"
                >Lighthouse({{ hover.lighthouse?.id }})</span
              >
              <div class="flex items-center gap-1 text-blue-300">
                <Icon.Zap class="w-3" />
                <span>{{ hover.lighthouse?.energy }}</span>
              </div>
            </div>
            <div class="flex items-center justify-center gap-2">
              <span
                v-for="player in hover.players"
                class="font-bold"
                :key="player.id"
                :class="[tailwind[player.id - 1]]">
                {{ player.name }}
              </span>
            </div>
          </div>
          <p v-else>Water tile</p>
        </div>
        <div v-else class="text-xs text-slate-500">
          Hover over a tile to see useful information
        </div>
      </div>
      <!-- Playback controls -->
    </div>
  </div>
</template>

<style src="@vueform/slider/themes/default.css"></style>
<style scoped>
  .list-move {
    transition: all 0.5s ease;
  }
</style>
