<template>
  <div ref="container" class="game-board">
  </div>
</template>

<script>
import * as pixi from 'pixi.js'
import { Board } from '@/code/presentation.js'
import { Game, Playback, PlaybackStatus } from '@/code/domain.js'

export default {
  name: 'GameBoard',
  props: {
    game: Game,
    playback: Playback,
  },
  async mounted() {
    const container = this.$refs.container;
    const app = new pixi.Application()
    await app.init({resizeTo: container});
    container.appendChild(app.canvas);

    const appWidth = app.renderer.width
    const appHeight = app.renderer.height
    const gridWidth = this.game.board.tiles[0].length;
    const gridHeight = this.game.board.tiles.length;
    const maxWidth = (appWidth - 100) / gridWidth;
    const maxHeight = (appHeight - 100) / gridHeight;
    const tileSize = Math.min(maxWidth, maxHeight);
    const x = appWidth / 2 - gridWidth * tileSize / 2;
    const y = appHeight / 2 - gridHeight * tileSize / 2;

    const board = new Board(this.game, tileSize, x, y);
    app.stage.addChild(board);

    const ticker = pixi.Ticker.shared;
    ticker.autoStart = false;
    this.playback.init(
      event => board.handle(event),
      () => ticker.start(),
      () => ticker.stop(),
    );

    ticker.add(() => {
      this.playback.auto() === PlaybackStatus.done && ticker.stop();
      console.log(ticker.started)
    })
  }
}
</script>

<style scoped>
.game-board {
  width: 100%;
  height: 100%;
  display: flex;
}
</style>
