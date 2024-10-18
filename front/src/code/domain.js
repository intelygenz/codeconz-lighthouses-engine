var clone = (object) => JSON.parse(JSON.stringify(object))

export class Position {
  constructor(x, y) {
    this.x = x
    this.y = y
  }
}

export class BoardItem {
  constructor(position) {
    this.position = position
  }

  get x() {
    return this.position.x
  }

  get y() {
    return this.position.y
  }
}

// ///////////////////////////////////////
// Player model
// ///////////////////////////////////////

export class Player extends BoardItem {
  constructor(id, position, energy, score, keys, name, color) {
    super(position)
    this.id = id
    this.name = name
    this.color = color
    this.energy = energy
    this.score = score
    this.keys = keys
  }
}

export class Lighthouse extends BoardItem {
  constructor(id, energy, ownerId, links, position) {
    super(position)
    this.id = id
    this.energy = energy
    this.ownerId = ownerId
    this.links = links
  }
}

export class Turn {
  constructor(player, lighthouses) {
    this.player = player
    this.lighthouses = lighthouses
  }
}

export class Setup {
  constructor(energy, players, lighthouses) {
    this.energy = energy
    this.players = players
    this.lighthouses = lighthouses
  }
}

export class Round {
  constructor(setup, turns) {
    this.setup = setup
    this.turns = turns
  }
}

export class Board {
  constructor(topology) {
    this.tiles = topology.map((row, y) => row.map((tileType, x) => 
      new Tile(new Position(x, y), new TileType(tileType))
    ))
  }
}

export class Tile extends BoardItem {
  constructor(position, type, energy) {
    super(position)
    this.type = type
    this.energy = energy
  }
}

export class TileType {
  constructor(type) {
    this.type = type
  }

  get isGround() {
    return this.type === 'g'
  }
}

export class Game {
  constructor(topology, setup, rounds) {
    this.board = new Board(topology)
    this.setup = setup
    this.rounds = rounds

    this.players = clone(setup.players)
    this.lighthouses = clone(setup.lighthouses)
  }
}

// ///////////////////////////////////////
// Playback model
// ///////////////////////////////////////

export class Frame extends Setup {
  constructor(name, energy, players, lighthouses) {
    super(energy, players, lighthouses)
    this.name = name
  }
}

export class GameStartFrame extends Frame {
  constructor(energy, players, lighthouses) {
    super("game.start", energy, players, lighthouses)
  }
}

export class RoundFrame extends Frame {
  constructor(roundIndex, energy, players, lighthouses) {
    super(`round-${roundIndex + 1}`, energy, players, lighthouses)
    this.roundIndex = roundIndex
  }

  get isRoundFrame() {
    return true
  }
}

export class TurnFrame extends Frame {
  constructor(roundIndex, turnIndex, playerName, energy, players, lighthouses) {
    super(`round-${roundIndex + 1}.turn-${turnIndex + 1}`, energy, players, lighthouses)
    this.roundIndex = roundIndex
    this.playerName = playerName
  }
}

export class Playback {
  constructor(game, speed) {
    this.game = game
    this.speed = speed
    this.tick = this.speed

    this.frameIndex = 0
    this.frames = [new GameStartFrame(game.setup.energy, game.setup.players, game.setup.lighthouses)]
    this.game.rounds.reduce((frames, round, roundIndex) => {
      frames.push(new RoundFrame(roundIndex, round.setup.energy, round.setup.players, round.setup.lighthouses))

      var players = round.setup.players
      round.turns.reduce((frames, turn, turnIndex) => {
        players = players.map(player => (player.id === turn.player.id ? turn.player : player))
        var playerName = this.game.players.find(p => p.id === turn.player.id).name
        frames.push(new TurnFrame(roundIndex, turnIndex, playerName, round.setup.energy, players, turn.lighthouses))
        return frames
      }, frames)
      return frames
    }, this.frames)
  }

  sync() {
    console.log(this.frames[0])
    this.game.players.forEach(player => {
      player.energy = this.currentFrame.players.find(p => p.id === player.id).energy
      player.score = this.currentFrame.players.find(p => p.id === player.id).score
    })
  }

  init(renderHandler, playHandler, pauseHandler) {
    this.render = renderHandler
    this.play = playHandler
    this.stop = pauseHandler
  }

  restart() {
    this.stop()
    this.frameIndex = 0
    this.render(this.currentFrame)
  }

  status() {
    if (--this.tick > 0) {
      return PlaybackStatus.playing
    }

    this.tick = this.speed
    return this.next();
  }

  next() {
    if (this.nextFrame) {
      this.sync()
      this.render(this.currentFrame)
      return PlaybackStatus.playing
    } else {
      return PlaybackStatus.done
    }
  }

  prev() {
    if (this.prevFrame) {
      this.sync()
      this.render(this.currentFrame)
      return PlaybackStatus.playing
    } else {
      return PlaybackStatus.done
    }
  }

  get isGameStart() {
    return this.frameIndex === 0
  }

  get hasGameEnded() {
    return this.frameIndex === this.frames.length - 1
  }

  get currentFrame() {
    return this.frames[this.frameIndex]
  }

  get nextFrame() {
    return this.hasGameEnded ? false : this.frames[++this.frameIndex]
  }

  get prevFrame() {
    return this.isGameStart ? false : this.frames[--this.frameIndex]
  }
}

export const PlaybackStatus = {
  playing: 'playing',
  done: 'done',
}

export const PlaybackEventType = {
  renderTurn: 'renderTurn',
  renderRound: 'renderRound',
}

export class PlaybackEvent {
  constructor(type, data) {
    this.type = type
    this.data = data
  }
}

export const renderTurnEvent = (data) => new PlaybackEvent(PlaybackEventType.renderTurn, data)
export const renderRoundEvent = (data) => new PlaybackEvent(PlaybackEventType.renderRound, data)

export const game = (...args) => new Game(...args)
export const player = (...args) => new Player(...args)
export const lighthouse = (...args) => new Lighthouse(...args)
export const round = (...args) => new Round(...args)
export const turn = (...args) => new Turn(...args)
export const setup = (...args) => new Setup(...args)
export const p = (...args) => new Position(...args)
