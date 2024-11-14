<script setup lang="ts">
  import {
    computed,
    ComputedRef,
    reactive,
    Ref,
    ref,
    useTemplateRef,
  } from "vue";
  import * as Icon from "lucide-vue-next";
  import { Stage, HoverTile } from "./code/presentation";
  import { EngineGame, mapGame } from "./code/engine_mapper";
  import {
    Playback,
    MaxSpeed,
    PlaybackStatus,
    HoverInfo,
  } from "./code/playback";
  import { Game, TileType } from "./code/domain";
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

  const tailwindFor = (id: number | undefined) =>
    id ? tailwind[id - 1] : "text-slate-100";

  var loadFile = ref(true);
  var game: Game = {} as Game;
  var stage: Stage = {} as Stage;
  var playback: Playback = {} as Playback;
  var status: Ref<PlaybackStatus> = {} as Ref<PlaybackStatus>;
  var hover: ComputedRef<HoverInfo> = {} as ComputedRef<HoverInfo>;

  const stageContainer = useTemplateRef("stageContainer");
  const hoverTile = reactive<HoverTile>({ x: undefined, y: undefined });

  const fileUploaded = async (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (!target.files || target.files.length == 0) return;

    const file = target.files[0];
    const reader = new FileReader();
    reader.onload = async () => {
      const json = JSON.parse(reader.result as string);
      game = mapGame(json as EngineGame);
      stage = new Stage(game, hoverTile);
      playback = new Playback(game, MaxSpeed);
      status = playback.status;
      hover = computed(() => status.value.hoverInfo(hoverTile));
      loadFile.value = false;
    };

    reader.readAsText(file);
  };

  const initStage = async () => {
    if (loadFile) {
      return;
    }
    if (!stageContainer.value) {
      console.error("Stage container not found");
      return;
    }
    stage.init(stageContainer.value, playback);
  };
</script>

<template>
  <div class="h-screen w-screen overflow-hidden bg-gray-950">
    <transition name="fade" mode="out-in" @after-enter="initStage">
      <div
        key="1"
        v-if="loadFile"
        class="flex h-full w-full items-center justify-center">
        <label
          for="file-dropzone"
          class="relative flex cursor-pointer flex-col items-center justify-center overflow-visible rounded-lg border-2 border-gray-600 bg-gray-500">
          <img
            alt="Lighthouse logo"
            src="@/assets/logo-sin-fondo.png"
            class="absolute -top-64 w-64" />
          <div
            class="flex flex-col items-center justify-center space-y-4 p-6 text-slate-200">
            <Icon.Upload class="w-8" />
            <p class="text-center">
              <span class="text-white">Upload</span> a JSON game
            </p>
          </div>
          <input
            id="file-dropzone"
            type="file"
            class="hidden"
            @change="fileUploaded" />
        </label>
      </div>

      <!-- Main -->
      <div v-else class="flex h-full w-full bg-black text-slate-100">
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
          <div ref="stageContainer" class="relative w-full grow"></div>
          <!-- Board -->

          <!-- Playback controls -->
          <div class="basis-1/5 space-y-2 text-center">
            <div class="flex justify-center gap-8 pb-2">
              <button @click="playback.restart" title="Restart">
                <Icon.RotateCcw />
                <!--<Icon.SkipBack />-->
              </button>
              <button @click="playback.prev" title="Previous Frame">
                <Icon.StepBack />
              </button>
              <button
                v-if="status.started"
                @click="playback.pause"
                title="Pause">
                <Icon.Pause />
              </button>
              <button v-else @click="playback.play" title="Play">
                <Icon.Play />
              </button>
              <button @click="playback.next" title="Next Frame">
                <Icon.StepForward />
              </button>
              <button title="Load new game" @click="loadFile = true">
                <Icon.Upload />
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
              <div v-if="hover.tile?.type == TileType.Ground">
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
                  <span :class="[tailwindFor(hover.lighthouse?.ownerId)]"
                    >Lighthouse({{ hover.lighthouse?.id }})</span
                  >
                  <div class="flex items-center gap-1 text-blue-300">
                    <Icon.Zap class="w-3" />
                    <span>{{ hover.lighthouse?.energy }}</span>
                  </div>
                </div>
                <div class="flex items-center justify-center gap-2 pt-1">
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
              Hover over a tile to check its data
            </div>
          </div>
          <!-- Playback controls -->
        </div>
      </div>
    </transition>
  </div>
</template>

<style src="@vueform/slider/themes/default.css"></style>
<style scoped>
  .list-move {
    transition: all 0.5s ease;
  }

  .fade-enter-active,
  .fade-leave-active {
    transition: opacity 0.5s ease;
  }

  .fade-enter-from,
  .fade-leave-to {
    opacity: 0;
  }
</style>
