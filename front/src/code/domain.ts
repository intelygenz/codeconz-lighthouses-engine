var clone = (object) => JSON.parse(JSON.stringify(object));

export class Position {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
}

export class BoardItem {
  constructor(position) {
    this.x = position.x;
    this.y = position.y;
  }
}

// ///////////////////////////////////////
// Player model
// ///////////////////////////////////////

export class Player extends BoardItem {
  constructor(id, position, energy, score, keys, name, color) {
    super(position);
    this.id = id;
    this.name = name;
    this.color = color;
    this.energy = energy;
    this.score = score;
    this.keys = keys;
  }
}

export class Lighthouse extends BoardItem {
  constructor(id, energy, ownerId, links, position) {
    super(position);
    this.id = id;
    this.energy = energy;
    this.ownerId = ownerId;
    this.links = links;
  }
}

export class Setup {
  constructor(energy, players, lighthouses) {
    this.energy = energy;
    this.players = players;
    this.lighthouses = lighthouses;
  }
}

export class Turn {
  constructor(player, lighthouses) {
    this.player = player;
    this.lighthouses = lighthouses;
  }

  get name() {
    return `End of ${this.player.name} turn`;
  }

  mergeSetup(setup) {
    return new Setup(
      setup.energy,
      setup.players.map((p) => (p.id === this.player.id ? this.player : p)),
      this.lighthouses,
    );
  }

  playersFor(players) {
    return;
  }
}

export class Round {
  constructor(setup, turns, index) {
    this.setup = setup;
    this.turns = turns;
    this.index = index;
  }

  pushFrames(frames) {
    frames.push(this.frame);

    var turnSetup = clone(this.setup);
    this.turns.forEach((turn) => {
      turnSetup = turn.mergeSetup(turnSetup);
      frames.push(new Frame(this.name, turn.name, turnSetup));
    });
  }

  get name() {
    return `Round ${this.index + 1}`;
  }

  get frame() {
    return new Frame(this.name, "Start", this.setup);
  }

  frameFor(turn) {
    return;
  }
}

export class Board {
  constructor(topology) {
    this.tiles = topology.map((row, y) =>
      row.map(
        (tileType, x) => new Tile(new Position(x, y), new TileType(tileType)),
      ),
    );
  }
}

export class Tile extends BoardItem {
  constructor(position, type, energy) {
    super(position);
    this.type = type;
    this.energy = energy;
  }
}

export class TileType {
  constructor(type) {
    this.type = type;
  }

  get isGround() {
    return this.type === "g";
  }
}

export class Game {
  constructor(topology, setup, rounds) {
    this.board = new Board(topology);
    this.setup = setup;
    this.rounds = rounds;
    this.players = clone(setup.players);
    this.lighthouses = clone(setup.lighthouses);

    this.rounds.forEach((round) => {
      round.turns.forEach((turn) => {
        turn.player.name = this.players.find(
          (p) => p.id == turn.player.id,
        ).name;
      });
    });
  }

  get frame() {
    return new Frame("Game Start", "", this.setup);
  }

  get orderedPlayers() {
    return this.players.slice().sort((a, b) => b.score - a.score);
  }
}

// ///////////////////////////////////////
// Playback model
// ///////////////////////////////////////

export class Frame extends Setup {
  constructor(title, subtitle, setup) {
    super(setup.energy, setup.players, setup.lighthouses);
    this.title = title;
    this.subtitle = subtitle;
  }
}

export class Playback {
  constructor(game, speed) {
    this.game = game;
    this.speed = speed;
    this.tick = this.speed;
    this.forward = true;

    this.frameIndex = 0;
    this.frames = [this.game.frame];
    this.game.rounds.forEach((round) => {
      round.pushFrames(this.frames);
    });
  }

  sync() {
    this.game.players.forEach((player) => {
      player.energy = this.currentFrame.players.find(
        (p) => p.id === player.id,
      ).energy;
      player.score = this.currentFrame.players.find(
        (p) => p.id === player.id,
      ).score;
    });
  }

  init(board, ticker) {
    this.board = board;
    this.render = (frame) => board.render(frame);
    this.play = () => ticker.start();
    this.stop = () => ticker.stop();
    ticker.add(() => {
      if (--this.tick > 0) {
        return;
      }

      this.tick = this.speed;
      if (!this.step()) {
        ticker.stop();
      }
    });
  }

  restart() {
    this.stop();
    this.frameIndex = 0;
    this.board.render(this.currentFrame);
  }

  status() {
    if (--this.tick > 0) {
      return PlaybackStatus.playing;
    }

    this.tick = this.speed;
    return this.next();
  }

  get isPlaying() {
    return this.ticker?.started;
  }

  step() {
    return this.forward ? this.next() : this.prev();
  }

  next() {
    if (this.nextFrame) {
      this.sync();
      this.board.render(this.currentFrame);
      return true;
    }
  }

  prev() {
    if (this.prevFrame) {
      this.sync();
      this.board.render(this.currentFrame);
      return true;
    }
  }

  get isGameStart() {
    return this.frameIndex === 0;
  }

  get hasGameEnded() {
    return this.frameIndex === this.frames.length - 1;
  }

  get currentFrame() {
    return this.frames[this.frameIndex];
  }

  get nextFrame() {
    return this.hasGameEnded ? false : this.frames[++this.frameIndex];
  }

  get prevFrame() {
    return this.isGameStart ? false : this.frames[--this.frameIndex];
  }
}

export const PlaybackStatus = {
  playing: "playing",
  done: "done",
};

export class PlaybackEvent {
  constructor(type, data) {
    this.type = type;
    this.data = data;
  }
}

export const game = (...args) => new Game(...args);
export const player = (...args) => new Player(...args);
export const lighthouse = (...args) => new Lighthouse(...args);
export const round = (...args) => new Round(...args);
export const turn = (...args) => new Turn(...args);
export const setup = (...args) => new Setup(...args);
export const p = (...args) => new Position(...args);
