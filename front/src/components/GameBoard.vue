<template>
  <div ref="container" class="game-board">
  </div>
</template>

<script>
import * as pixi from 'pixi.js';
import water from '@/assets/Water+.png';

class Tile extends pixi.Container {
  constructor(x, y, width, height, sprite) {
    super()
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.sprite = sprite;

    this.addChild(sprite);
  }
}

class Board extends pixi.Container {
  constructor(width, height, tiles, sprite) {
    super()
    this.tiles = tiles;
    this.width = width;
    this.height = height;

    let tileWidth = this.width / this.tiles[0].length;
    let tileHeight = this.height / this.tiles.length;

    this.tiles.forEach((row, rowIndex) => {
      row.forEach((col, colIndex) => {
        let x = rowIndex * tileWidth;
        let y = colIndex * tileHeight;
        let tile = new Tile(x, y, tileWidth, tileHeight, sprite);
        this.addChild(tile);
      });
    });
  }
}

export default {
  name: 'GameBoard',
  props: {
    tiles: Array
  },
  async mounted() {
    const container = this.$refs.container;

    const app = new pixi.Application()
    await app.init({resizeTo: container});
    app.renderer.backgroundColor = 0x000000;
    container.appendChild(app.canvas);

    await pixi.Assets.load(water);
    const spritesheet = new pixi.Spritesheet(
        pixi.Texture.from(water),
        {
          frames: {
            water: {
              frame: {x: 0, y: 0, w: 16, h: 16},
              spriteSourceSize: {x: 0, y: 0, w: 16, h: 16},
              sourceSize: {w: 16, h: 16},
              anchor: {x: 0, y: 0}
            }
          },
          meta: {
            image: "assets/Water+.png",
            size: {w: 14*16, h: 12*16}
          }
        }
    );
    await spritesheet.parse();

    const sprite = new pixi.Sprite(spritesheet.textures.water);
    console.log(sprite)

    const board = new Board(800, 800, this.tiles, sprite);
    app.stage.addChild(board);
  },
}
</script>

<style scoped>
.game-board {
  width: 100%;
  height: 100%;
  display: flex;
}
</style>
