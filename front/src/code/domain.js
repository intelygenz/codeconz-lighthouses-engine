export class Coordinates {
  constructor(x, y) {
    this.x = x
    this.y = y
  }

  equals(other) {
    return this.x === other.x && this.y === other.y
  }
}

export class Positionable {
  constructor(coordinates) {
    this.coordinates = coordinates
  }

  syncCoordinates(coordinates) {
    this.coordinates = coordinates
  }

  get x() {
    return this.coordinates.x
  }

  get y() {
    return this.coordinates.y
  }
}

export class Game {
  constructor(map, players, lighthouses, rounds) {
    this.board = new Board(map)

    this.players = players.sort((a, b) => a.id - b.id)
    this.players.forEach(player => {
      this.board.tileFor(player).players.push(player)
    })

    this.lighthouses = lighthouses.sort((a, b) => a.id - b.id)
    this.lighthouses.forEach(lighthouse => {
      this.board.tileFor(lighthouse).lighthouse = lighthouse
    })

    this.rounds = rounds.sort((a, b) => a.id - b.id)
  }
}

export class Board {
  constructor(map) {
    this.tiles = map.map((row, y) => row.map((tileType, x) => 
      new Tile(new Coordinates(x, y), new TileType(tileType), null)
    ))
  }

  tileFor(positionable) {
    return this.tiles[positionable.y][positionable.x]
  }
}

export class BoardStatus {
  constructor(energyMap) {
    this.energyMap = energyMap
  }
}

export class TileType {
  constructor(tileType) {
    this.tileType = tileType
  }

  get isGround() {
    return this.tileType === "g"
  }
}

export class Tile extends Positionable {
  constructor(coordinates, type, energy) {
    super(coordinates)
    this.type = type
    this.energy = energy
    this.players = []
  }
}

export class Player extends Positionable {
  constructor(id, name, color, coordinates) {
    super(coordinates)
    this.id = id
    this.index = id - 1
    this.name = name
    this.color = color
    this.energy = 0
    this.score = 0
    this.keys = []

    this.turnIndex = 0
    this.turns = [new Turn(new PlayerStatus(this.id, 0, []), this.coordinates)]
  }

  get currentTurn() {
    return this.turns[this.turnIndex]
  }

  advance() {
    if (this.turnIndex < this.turns.length) {
      this.turnIndex++
      this.syncCurrent()
      return playerMoveEvent(this.currentTurn)
    }
  }

  rewind() {
    if (this.turnIndex > 0) {
      this.turnIndex--
      this.syncCurrent()
      return playerMoveEvent(this.currentTurn)
    }
  }

  syncCurrent() {
    this.syncStatus(this.currentTurn.playerStatus)
    this.syncCoordinates(this.currentTurn.coordinates)
  }

  syncStatus(playerStatus) {
    this.energy = playerStatus.energy
    this.keys = playerStatus.keys
  }

  syncScore(playerScore) {
    this.score = playerScore.score
  }
}

export class Lighthouse extends Positionable {
  constructor(id, coordinates) {
    super(coordinates)
    this.id = id
    this.energy = 0
    this.ownerId = null
    this.connections = []
  }

  syncStatus(lighthouseStatus) {
    this.energy = lighthouseStatus.energy
    this.ownerId = lighthouseStatus.ownerId
    this.connections = lighthouseStatus.connections
  }
}

export class Round {
  constructor(id, initialRoundStatus, turns) {
    this.id = id
    this.index = id - 1
    this.initialStatus = initialRoundStatus
    this.turns = turns
  }

  syncData(players, lighthouses) {
    this.initialStatus.syncData(players, lighthouses)
  }
}

export class InitialRoundStatus {
  constructor(boardStatus, playerStatuses, playerScores, lighthouseStatuses) {
    this.boardStatus = boardStatus
    this.playerStatuses = playerStatuses
    this.playerScores = playerScores
    this.lighthouseStatuses = lighthouseStatuses
  }

  playerScore(playerId) {
    return this.playerScores.find(score => score.playerId === playerId)
  }

  playerStatus(playerId) {
    return this.playerStatuses.find(status => status.playerId === playerId)
  }

  lighthouseStatus(lighthouseId) {
    return this.lighthouseStatuses.find(status => status.lighthouseId === lighthouseId)
  }

  syncData(players, lighthouses) {
    players.forEach(player => {
      player.syncScore(this.playerScore(player.id))
      player.syncStatus(this.playerStatus(player.id))
    })
    lighthouses.forEach(lighthouse => 
      lighthouse.syncStatus(this.lighthouseStatus(lighthouse.id))
    )
  }
}

export class Turn {
  constructor(playerStatus, coordinates, lighthouseStatus) {
    this.playerStatus = playerStatus
    this.coordinates = coordinates
    this.lighthouseStatus = lighthouseStatus
  }

  get playerId() {
    return this.playerStatus.playerId
  }
}

export class PlayerStatus {
  constructor(playerId, energy, keys) {
    this.playerId = playerId
    this.energy = energy
    this.keys = keys
  }
}

export class PlayerScore {
  constructor(playerId, score) {
    this.playerId = playerId
    this.score = score
  }
}

export class LighthouseStatus {
  constructor(lighthouseId, ownerId, energy, connections) {
    this.lighthouseId = lighthouseId
    this.ownerId = ownerId
    this.energy = energy
    this.connections = connections
  }
}

const NO_ROUND = { isStub: true, syncData: () => {} }
const NO_PLAYER = { isStub: true, advance: () => {}, rewind: () => {} }

export class Playback {
  constructor(game, speed) {
    this.game = game
    this.speed = speed
    this.game.rounds.forEach(round => round.turns.forEach(turn => {
      this.game.players.find(player => player.id === turn.playerId).turns.push(turn)
    }))

    this.gameStart = new InitialRoundStatus(
      new BoardStatus(this.game.board.tiles.map(row => row.map(() => 0))),
      this.game.players.map(player => new PlayerStatus(player.id, 0, [])),
      this.game.players.map(player => new PlayerScore(player.id, 0)),
      this.game.lighthouses.map(lighthouse => new LighthouseStatus(lighthouse.id, null, 0, []))
    )

    this.currentRound = NO_ROUND
    this.currentPlayer = NO_PLAYER
    this.currentFrame = this.speed
  }

  init(eventHandler, playHandler, pauseHandler) {
    this.handle = eventHandler
    this.playHandler = playHandler
    this.pauseHandler = pauseHandler
    this.reset()
  }

  reset() {
    this.currentRound = NO_ROUND
    this.currentPlayer = NO_PLAYER
    this.currentFrame = this.speed
    this.syncCurrentRound()
    this.handle(gameStartEvent(this.gameStart))
  }

  play() {
    this.playHandler()
  }

  pause() {
    this.pauseHandler()
  }

  auto() {
    if (--this.currentFrame > 0) {
      return PlaybackStatus.playing
    }

    this.currentFrame = this.speed
    return this.next();
  }

  next() {
    if (this.currentRound.isStub) {
      this.currentRound = this.rounds[0]
      this.syncCurrentRound()
      this.handle(roundStartEvent(this.currentRound.initialStatus))
      return PlaybackStatus.playing
    }

    let event = this.nextPlayer?.advance()
    if (event) {
      this.currentPlayer = this.nextPlayer
      this.handle(event)
      return PlaybackStatus.playing
    } else if (this.nextRound) {
      this.currentRound = this.nextRound
      this.currentPlayer = NO_PLAYER
      this.syncCurrentRound()
      this.handle(roundStartEvent(this.currentRound.initialStatus))
      return PlaybackStatus.playing
    } else {
      return PlaybackStatus.done
    }
  }

  prev() {
    this.currentFrame = this.speed

    let event = this.currentPlayer.rewind()
    if (event) {
      this.currentPlayer = this.prevPlayer
      this.handle(event)
      return PlaybackStatus.playing
    } else if (this.prevRound) {
      this.currentRound = this.prevRound
      this.currentPlayer = this.lastPlayer
      this.syncCurrentRound()
      this.handle(roundStartEvent(this.currentRound.initialStatus))
      return PlaybackStatus.playing
    } else if (!this.currentRound.isStub) {
      this.currentRound = NO_ROUND
      this.syncCurrentRound()
      this.handle(gameStartEvent(this.gameStart))
      return PlaybackStatus.playing
    } else {
      return PlaybackStatus.done
    }
  }

  syncCurrentRound() {
    this.currentRound.syncData(this.game.players, this.game.lighthouses)
  }

  get lastPlayer() {
    return this.players[this.players.length - 1]
  }

  get nextPlayer() {
    return this.players[this.currentPlayer.isStub ? 0 : this.currentPlayer.index + 1]
  }

  get prevPlayer() {
    return this.players[this.currentPlayer.index - 1] || NO_PLAYER
  }

  get nextRound() {
    return this.rounds[this.currentRound.index + 1]
  }

  get prevRound() {
    return this.rounds[this.currentRound.index - 1]
  }

  get players() {
    return this.game.players
  }

  get rounds() {
    return this.game.rounds
  }
}

export const PlaybackStatus = {
  playing: 'playing',
  done: 'done',
}

export const PlaybackEventType = {
  gameStart: 'gameStart',
  roundStart: 'roundStart',
  playerMove: 'playerMove',
}

export class PlaybackEvent {
  constructor(type, data) {
    this.type = type
    this.data = data
  }
}

export const gameStartEvent = (data) => new PlaybackEvent(PlaybackEventType.gameStart, data)
export const roundStartEvent = (data) => new PlaybackEvent(PlaybackEventType.roundStart, data)
export const playerMoveEvent = (data) => new PlaybackEvent(PlaybackEventType.playerMove, data)

export const game = (...args) => new Game(...args)
export const player = (...args) => new Player(...args)
export const lighthouse = (...args) => new Lighthouse(...args)
export const round = (...args) => new Round(...args)
export const turn = (...args) => new Turn(...args)
export const initialRoundStatus = (...args) => new InitialRoundStatus(...args)
export const boardStatus = (...args) => new BoardStatus(...args)
export const playerStatus = (...args) => new PlayerStatus(...args)
export const playerScore = (...args) => new PlayerScore(...args)
export const lighthouseStatus = (...args) => new LighthouseStatus(...args)
export const c = (...args) => new Coordinates(...args)
