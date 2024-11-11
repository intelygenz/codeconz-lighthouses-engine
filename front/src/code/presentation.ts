import * as pixi from "pixi.js";
import { PlaybackStatus } from "@/code/domain";

class Tile extends pixi.Container {
  constructor(type, size, x, y) {
    super();
    this.label = `Tile(${x}, ${y})`;
    this.type = type;
    this.size = size;
    this.position.set(x * size, y * size);

    if (this.type.isGround) {
      this.ground = new IslandBackground(size, x, y);
      this.addChild(this.ground);
      this.addChild(new Marker(size));
    } else {
      this.addChild(new WaterBackground(size, x, y));
    }

    this.players = new Players(this.label, this.size);
    this.addChild(this.players);
    this.addChild(this.players.mask);
    this.energy = 0;

    this.interactive = true;
    this.onmouseover = (event) => {
      console.log(this.label);
    };
  }

  move(player) {
    this.players.move(player);
  }

  set energy(energy) {
    if (this.type.isGround) {
      this.ground.energy = energy;
    }
  }
}

class Players extends pixi.Container {
  constructor(label, size) {
    super();
    this.label = `PlayersContainer(${label})`;
    this.mask = new Player("mask", 0, 0xffffff, size);
    this.sortableChildren = true;
  }

  move(player) {
    if (player.parent) {
      const originalParent = player.parent;
      originalParent.removeChild(player);
      originalParent.accomodate();
    }

    this.addChild(player);
    this.accomodate();
  }

  accomodate() {
    const children = this.children.filter((child) => child instanceof Player);
    children
      .sort((a, b) => a.zIndex - b.zIndex)
      .forEach((player, index) => {
        player.position.x =
          player.parentSize * 0.5 + (index * player.size) / children.length;
      });
  }
}

class Player extends pixi.Graphics {
  constructor(id, index, color, parentSize) {
    super();
    this.label = `Player(${id})`;
    this.color = color;
    this.size = parentSize * 0.75;
    this.parentSize = parentSize;

    this.rect(0, 0, this.size, this.size).fill({ color: this.color });
    this.pivot.set(this.size * 0.5);
    this.position.set(this.parentSize * 0.5);
    this.zIndex = index;

    const colorMatrix = new pixi.ColorMatrixFilter();
    colorMatrix.brightness(0.7);
    this.filters = [colorMatrix];
  }
}

class Lighthouse extends pixi.Graphics {
  constructor(data, parentSize) {
    super();
    this.size = parentSize * 0.4;

    this.label = `Lighthouse(${data.id})`;
    this.position.set(parentSize * 0.5);
    this.pivot.set(this.size * 0.5);
    this.rotation = Math.PI / 4;
    this.rect(0, 0, this.size, this.size).fill({ color: 0xf0f0f0 });

    this.colorMatrix = new pixi.ColorMatrixFilter();
    this.filters = [this.colorMatrix];
  }

  update(lighthouse, owner) {
    this.colorMatrix.reset();
    if (owner) {
      this.tint = owner.color;
      // this.colorMatrix.saturate(.9)
      // this.colorMatrix.greyscale(.5)
    } else {
      this.tint = 0xffffff;
    }

    lighthouse.links.forEach((link) => {});
  }
}

class WaterBackground extends pixi.Graphics {
  constructor(parentSize) {
    super();
    this.label = "WaterBackground";
    this.rect(0, 0, parentSize, parentSize).fill({ color: 0x000000 });
  }
}

class IslandBackground extends pixi.Graphics {
  constructor(parentSize) {
    super();
    this.label = "IslandBackground";
    this.rect(0, 0, parentSize, parentSize).fill({ color: 0x606060 });
  }

  set energy(energy) {
    this.tint = 0x606060 + Math.round(energy * 0.75) * 0x010101;
  }
}

class Marker extends pixi.Graphics {
  constructor(parentSize) {
    super();
    this.label = "Marker";
    this.position.set(parentSize * 0.5);
    this.rect(0, 0, 1, 1).fill({ color: 0xf0f0f0 });
  }
}

export class Board extends pixi.Container {
  constructor(game, tileSize, x, y) {
    super();
    this.tileSize = tileSize;
    this.position.set(x, y);

    this.tiles = game.board.tiles.map((row, y) =>
      row.map((gameTile, x) => {
        const tile = new Tile(gameTile.type, this.tileSize, x, y);
        this.addChild(tile);
        return tile;
      }),
    );

    this.players = game.setup.players.reduce((players, player, index) => {
      players[player.id] = new Player(
        player.id,
        index,
        player.color,
        this.tileSize,
      );

      this.tiles[player.y][player.x].move(players[player.id]);
      return players;
    }, {});

    this.lighthouses = game.setup.lighthouses.reduce(
      (lighthouses, lighthouse) => {
        lighthouses[lighthouse.id] = new Lighthouse(lighthouse, this.tileSize);
        lighthouses[lighthouse.id].update(
          lighthouse,
          this.players[lighthouse.ownerId],
        );
        this.tiles[lighthouse.y][lighthouse.x].addChild(
          lighthouses[lighthouse.id],
        );
        return lighthouses;
      },
      {},
    );

    game.setup.lighthouses.forEach((origin) => {
      game.setup.lighthouses.forEach((target) => {
        const link = new pixi.Graphics();
        const from = this.lighthouses[origin.id];
        const to = this.lighthouses[target.id];
        link.moveTo(from.parent.x + tileSize / 2, from.parent.y + tileSize / 2);
        link.lineTo(to.parent.x + tileSize / 2, to.parent.y + tileSize / 2);
        link.stroke({ width: 1 });
        link.filters = [new pixi.AlphaFilter({ alpha: 1 })];
        link.tint = 0x1f22f8;
        // link.alpha = -1;
        this.addChild(link);
        // link.filters = [];
        link.tint = 0x22f81f;
        // link.alpha = 1;
      });
    });
  }

  render(frame) {
    frame.energy.forEach((row, y) =>
      row.forEach((energy, x) => {
        this.tiles[y][x].energy = energy;
      }),
    );

    frame.players.forEach((player) => {
      this.tiles[player.y][player.x].move(this.players[player.id]);
    });

    frame.lighthouses.forEach((lighthouse) => {
      this.lighthouses[lighthouse.id].update(
        lighthouse,
        this.players[lighthouse.ownerId],
      );

      lighthouse.links.forEach((targetId) => {
        const link = new pixi.Graphics();
        const origin = this.lighthouses[lighthouse.id];
        const target = this.lighthouses[targetId];
        link.moveTo(origin.parent.x, origin.parent.y);
        link.lineTo(target.parent.x, target.parent.y);
        link.stroke({ color: 0x606060, width: 1 });
        this.addChild(link);
        link.zIndex = 1;
      });
    });
  }
}

export const init = async (game, playback, container) => {
  const app = new pixi.Application();
  await app.init({ resizeTo: container });
  container.appendChild(app.canvas);

  const appWidth = app.renderer.width;
  const appHeight = app.renderer.height;

  const gridWidth = game.board.tiles[0].length;
  const gridHeight = game.board.tiles.length;
  const maxWidth = (appWidth - 100) / gridWidth;
  const maxHeight = (appHeight - 100) / gridHeight;
  const tileSize = Math.min(maxWidth, maxHeight);
  const x = appWidth / 2 - (gridWidth * tileSize) / 2;
  const y = appHeight / 2 - (gridHeight * tileSize) / 2;

  const board = new Board(game, tileSize, x, y);
  app.stage.addChild(board);

  const border = new pixi.Graphics();
  border
    .rect(x, y, tileSize * gridWidth, tileSize * gridHeight)
    .stroke({ color: 0x606060, width: 1 });
  app.stage.addChild(border);

  const ticker = pixi.Ticker.shared;
  ticker.autoStart = false;

  return { board, ticker };
};
