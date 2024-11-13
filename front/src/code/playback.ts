import { Ticker } from "pixi.js";
import { Board, Game, Lighthouse, Player, State, TileType } from "./domain";
import { HoverTile, Tileset } from "./presentation";
import { Ref, ref } from "vue";

export const MaxSpeed: number = 5;
const MaxTicks: number = MaxSpeed * 2 + 1;

export interface HoverInfo {
  show: boolean;
  tile: TileInfo | undefined;
  lighthouse: LighthouseInfo | undefined;
  players: Array<PlayerInfo>;
}

export interface TileInfo {
  x: number;
  y: number;
  type: TileType;
  energy: number;
}

export interface LighthouseInfo {
  id: number;
  energy: number;
  ownerId: number | undefined;
}

export interface PlayerInfo {
  id: number;
  name: string;
  keys: Array<number>;
}

export interface Frame extends State {
  title: string;
  subtitle: string;
}

export class PlaybackStatus implements Frame {
  started: boolean;
  title: string;
  subtitle: string;
  energy: Array<Array<number>>;
  players: Array<Player>;
  lighthouses: Array<Lighthouse>;

  constructor(
    frame: Frame,
    public board: Board,
  ) {
    this.started = false;
    this.title = frame.title;
    this.subtitle = frame.subtitle;
    this.energy = frame.energy;
    this.players = frame.players;
    this.lighthouses = frame.lighthouses;
  }

  merge(frame: Frame) {
    this.energy = frame.energy;
    frame.players.forEach((player) => {
      const statusPlayer = this.players.find(
        (p) => p.id == player.id,
      ) as Player;
      statusPlayer.energy = player.energy;
      statusPlayer.score = player.score;
    });
    this.lighthouses = frame.lighthouses;
  }

  get scoreboard(): Array<Player> {
    return this.players.sort((a, b) => b.score - a.score);
  }

  hoverInfo(tile: HoverTile): HoverInfo {
    if (tile.x === undefined || tile.y === undefined) {
      return {
        show: false,
        tile: undefined,
        lighthouse: undefined,
        players: [],
      };
    }

    return {
      show: true,
      tile: this.tileInfo(tile.x, tile.y),
      lighthouse: this.lighthouseInfo(tile.x, tile.y),
      players: this.playersInfo(tile.x, tile.y),
    };
  }

  tileInfo(x: number, y: number): TileInfo {
    return {
      x,
      y,
      type: this.board[y][x].type,
      energy: this.energy[y][x],
    };
  }

  lighthouseInfo(x: number, y: number): LighthouseInfo | undefined {
    const lighthouse = this.lighthouses.find((lh) => lh.x == x && lh.y == y);
    if (!lighthouse) {
      return;
    }

    return {
      id: lighthouse.id,
      energy: lighthouse.energy,
      ownerId: lighthouse.ownerId,
    };
  }

  playersInfo(x: number, y: number): Array<PlayerInfo> {
    return this.players
      .filter((p) => p.x === x && p.y === y)
      .map((p) => ({
        id: p.id,
        name: p.name,
        keys: p.keys,
      }));
  }
}

export const frame = (
  title: string,
  subtitle: string,
  state: State,
): Frame => ({
  title,
  subtitle,
  energy: state.energy,
  players: state.players,
  lighthouses: state.lighthouses,
});

export class Playback {
  public tick: number;
  public cursor: number;
  public frames: Array<Frame>;
  public forward: boolean = true;
  public tileset: Tileset | undefined;
  public ticker: Ticker | undefined;
  public status: Ref<PlaybackStatus>;

  constructor(
    public game: Game,
    public speed: number,
  ) {
    this.tick = this.speed;
    this.cursor = 0;
    this.frames = [
      frame("Game Start", "", this.game.state),
      ...this.game.rounds.flatMap((round) => [
        frame(round.name, "Start", round.state),
        ...round.turns.map((turn) => frame(round.name, turn.name, turn.state)),
      ]),
      frame("Game End", "", this.game.finalState),
    ];

    this.status = ref(new PlaybackStatus(this.frames[0], this.game.board));
  }

  init(tileset: Tileset, ticker: Ticker) {
    this.tileset = tileset;
    this.ticker = ticker;

    this.ticker.add(() => {
      this.status.value.started = true;
      if (--this.tick > 0) {
        return;
      }

      this.tick = MaxTicks - this.speed * 2;
      if (!this.next()) {
        this.ticker?.stop();
        this.status.value.started = false;
      }
    });
  }

  get scoreboard() {
    return this.currFrame.players.sort((a, b) => b.score - a.score);
  }

  set frame(value: number) {
    this.pause();
    this.cursor = value;
    this.sync();
  }

  get frame() {
    return this.cursor;
  }

  restart() {
    this.cursor = 0;
    this.sync();
    this.pause();
  }

  play() {
    this.status.value.started = true;
    this.ticker?.start();
  }

  pause() {
    this.status.value.started = false;
    this.ticker?.stop();
  }

  next() {
    return this.nextFrame ? this.sync() : false;
  }

  prev() {
    return this.prevFrame ? this.sync() : false;
  }

  sync() {
    Object.assign(this.status.value, this.currFrame);
    return this.tileset?.render(this.currFrame);
  }

  get currFrame() {
    return this.frames[this.cursor];
  }

  get nextFrame() {
    return this.isEnd ? false : this.frames[++this.cursor];
  }

  get prevFrame() {
    return this.isStart ? false : this.frames[--this.cursor];
  }

  get isStart() {
    return this.cursor === 0;
  }

  get isEnd() {
    return this.cursor === this.frames.length - 1;
  }
}
