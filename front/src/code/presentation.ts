import {
  Game,
  Lighthouse as LighthouseData,
  Tile as TileData,
  Player as PlayerData,
  TileType,
} from "./domain";
import {
  Application,
  ColorMatrixFilter,
  Container,
  Graphics,
  Ticker,
} from "pixi.js";
import { Frame, Playback } from "./playback";

export class Stage {
  constructor(
    public game: Game,
    public hover: HoverTile,
  ) {}

  async init(container: HTMLElement, playback: Playback) {
    const app = new Application();
    await app.init({ resizeTo: container });
    container.appendChild(app.canvas);

    const width = this.game.board[0].length;
    const height = this.game.board.length;
    const maxTileWidth = (app.renderer.width - 50) / width;
    const maxTileHeight = (app.renderer.height - 50) / height;
    const tileSize = Math.min(maxTileWidth, maxTileHeight);
    const x = app.renderer.width / 2 - (width * tileSize) / 2;
    const y = app.renderer.height / 2 - (height * tileSize) / 2;

    const tileset = new Tileset(this.game, tileSize, this.hover, x, y);
    app.stage.addChild(tileset);

    const border = new Graphics();
    border
      .rect(x, y, tileSize * width, tileSize * height)
      .stroke({ color: 0x606060, width: 1 });
    app.stage.addChild(border);

    const ticker = Ticker.shared;
    ticker.autoStart = false;
    playback.init(tileset, ticker);
  }
}

export class Tileset extends Container {
  public tiles: Array<Array<Tile>>;
  public players: Map<number, Player>;
  public lighthouses: Map<number, Lighthouse>;
  public links: Map<string, Graphics> = new Map();

  constructor(
    public game: Game,
    public tileSize: number,
    hover: HoverTile,
    x: number,
    y: number,
  ) {
    super();
    this.position.set(x, y);

    this.tiles = game.board.map((row) =>
      row.map((gameTile) => {
        const tile = new Tile(gameTile, this.tileSize, hover);
        this.addChild(tile);
        return tile;
      }),
    );

    this.players = game.state.players.reduce((players, playerData) => {
      const player = new Player(playerData, this.tileSize);
      this.tiles[playerData.y][playerData.x].move(player);

      players.set(playerData.id, player);
      return players;
    }, new Map());

    this.lighthouses = game.state.lighthouses.reduce((lh, lhData) => {
      const lighthouse = new Lighthouse(lhData, this.tileSize);
      this.tiles[lhData.y][lhData.x].lighthouse = lighthouse;

      lh.set(lhData.id, lighthouse);
      return lh;
    }, new Map());
  }

  render(frame: Frame) {
    frame.energy.forEach((row, y) =>
      row.forEach((energy, x) => {
        this.tiles[y][x].setEnergy(energy);
      }),
    );

    frame.players.forEach((playerData) => {
      const player = this.players.get(playerData.id) as Player;
      this.tiles[playerData.y][playerData.x].move(player);
    });

    const frameLinks = new Array<string>();
    frame.lighthouses.forEach((lighthouseData) => {
      const lighthouse = this.lighthouses.get(lighthouseData.id) as Lighthouse;
      const owner = this.players.get(lighthouseData.ownerId);
      lighthouse.update(owner);

      lighthouseData.links.forEach((targetId) => {
        if (owner === undefined) {
          console.warn("Lighthouse has links but no owner", lighthouseData);
          return;
        }

        const linkKey = keyFor(lighthouseData.id, targetId);
        frameLinks.push(linkKey);
        if (!this.links.has(linkKey)) {
          this.registerLink(linkKey, lighthouseData.id, targetId, owner.color);
        }
      });
    });

    for (const [key, link] of this.links) {
      if (!frameLinks.includes(key)) {
        this.removeChild(link);
        this.links.delete(key);
      }
    }

    return true;
  }

  registerLink(linkKey: string, origin: number, target: number, color: number) {
    if (this.links.get(linkKey)) {
      return;
    }

    const from = this.lighthouses.get(origin) as Lighthouse;
    const to = this.lighthouses.get(target) as Lighthouse;

    const link = new Graphics();
    link.moveTo(
      from.parent.x + this.tileSize / 2,
      from.parent.y + this.tileSize / 2,
    );
    link.lineTo(
      to.parent.x + this.tileSize / 2,
      to.parent.y + this.tileSize / 2,
    );
    link.stroke({ color, width: 1 });
    // link.filters = [];
    // link.tint = color;

    this.addChild(link);
    this.links.set(linkKey, link);
    // link.filters = [];
    // link.tint = 0x22f81f;
  }
}

const keyFor = (origin: number, target: number) =>
  origin < target ? `${origin},${target}` : `${target},${origin}`;

class Tile extends Container {
  public ground: IslandBackground | null;
  public players: Players;
  public lighthouseChild: Lighthouse | undefined;

  constructor(
    public data: TileData,
    public size: number,
    public hover: HoverTile,
  ) {
    super();
    this.position.set(data.x * size, data.y * size);

    this.ground = null;
    if (data.type == TileType.Ground) {
      this.ground = new IslandBackground(size);
      this.addChild(this.ground);
      this.addChild(new Marker(size));
    } else {
      this.addChild(new WaterBackground(size));
    }

    this.players = new Players(this.size);
    this.addChild(this.players);
    this.addChild(this.players.mask as Container);

    this.on("mouseenter", this.buildHover.bind(this));
    this.on("mouseleave", this.clearHover.bind(this));
    this.eventMode = "static";
  }

  move(player: Player) {
    this.players.move(player);
  }

  set lighthouse(lighthouse: Lighthouse) {
    this.lighthouseChild = lighthouse;
    this.addChild(lighthouse);
  }

  setEnergy(energy: number) {
    if (this.ground) {
      if (energy !== this.ground.energy) {
        this.data.energy = energy;
        this.ground.setEnergy(energy);
      }
    }
  }

  buildHover() {
    this.hover.x = this.data.x;
    this.hover.y = this.data.y;
  }

  clearHover() {
    delete this.hover.x;
    delete this.hover.y;
  }
}

export interface HoverTile {
  x: number | undefined;
  y: number | undefined;
}

class Players extends Container {
  constructor(size: number) {
    super();
    this.mask = new Player(null, size);
    this.sortableChildren = true;
  }

  move(player: Player) {
    if (player.parent) {
      const players = player.parent as Players;
      players.removeChild(player);
      players.accomodate();
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

class Player extends Graphics {
  public size: number;
  public color: number;

  constructor(
    data: PlayerData | null,
    public parentSize: number,
  ) {
    super();
    this.size = parentSize * 0.75;
    this.color = data ? data.color : 0xffffff;

    this.rect(0, 0, this.size, this.size).fill({ color: this.color });
    this.pivot.set(this.size * 0.5);
    this.position.set(this.parentSize * 0.5);

    const colorMatrix = new ColorMatrixFilter();
    colorMatrix.brightness(0.7, true);
    this.filters = [colorMatrix];
  }
}

class Lighthouse extends Graphics {
  public ownedBy: Player | undefined;

  constructor(data: LighthouseData, parentSize: number) {
    super();
    const size = parentSize * 0.4;

    this.label = `Lighthouse(${data.id})`;
    this.position.set(parentSize * 0.5);
    this.rotation = Math.PI / 4;
    this.pivot.set(size * 0.5);
    this.rect(0, 0, size, size).fill({ color: 0xf0f0f0 });
  }

  update(owner: Player | undefined) {
    this.tint = owner ? owner.color : 0xffffff;
  }
}

class WaterBackground extends Graphics {
  constructor(parentSize: number) {
    super();
    this.label = "WaterBackground";
    this.rect(0, 0, parentSize, parentSize).fill({ color: 0x000000 });
  }
}

class IslandBackground extends Graphics {
  public energy: number;

  constructor(public parentSize: number) {
    super();
    this.label = "IslandBackground";
    this.energy = 0;
    this.rect(0, 0, parentSize, parentSize).fill({ color: 0x606060 });
    this.setEnergy(0);
  }

  setEnergy(energy: number) {
    this.energy = energy;
    this.tint = 0x606060 + Math.round(energy * 0.75) * 0x010101;
  }
}

class Marker extends Graphics {
  constructor(parentSize: number) {
    super();
    this.label = "Marker";
    this.position.set(parentSize * 0.5);
    this.rect(0, 0, 1, 1).fill({ color: 0xf0f0f0 });
  }
}

// status() {
//   if (--this.tick > 0) {
//     return PlaybackStatus.playing;
//   }
//
//   this.tick = this.speed;
//   return this.next();
// }
