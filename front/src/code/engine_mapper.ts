import {
  game,
  p,
  tile,
  state,
  TileType,
  round,
  turn,
  State,
  player,
  lighthouse,
} from "./domain";
import { colorFor } from "./palette";

export interface EngineGame {
  topology: Array<Array<boolean>>;
  setup: EngineState;
  rounds: Array<EngineRound>;
  finalStatus: EngineState;
}

interface EngineState {
  energy: Array<Array<number>>;
  players: Array<EnginePlayer>;
  lighthouses: Array<EngineLighthouse>;
}

interface EnginePlayer {
  id: number;
  name: string;
  score: number;
  energy: number;
  position: Array<number>;
  keys: Array<number>;
}

interface EngineLighthouse {
  id: number;
  energy: number;
  ownerId: number;
  position: Array<number>;
  connections: Array<number>;
}

interface EngineRound {
  setup: EngineState;
  turns: Array<EngineTurn>;
}

interface EngineTurn {
  player: EnginePlayer;
  lighthouses: Array<EngineLighthouse>;
}

export const mapGame = (engineGame: EngineGame) =>
  game(
    mapBoard(engineGame.topology),
    mapState(engineGame.setup),
    engineGame.rounds.map(mapRound),
    mapState(engineGame.finalStatus),
  );

const mapBoard = (topology: Array<Array<boolean>>) =>
  topology.map((row, rowIndex) =>
    row.map((tileType, colIndex) =>
      tile(mapTileType(tileType), 0, mapPosition([rowIndex, colIndex])),
    ),
  );

const mapTileType = (tileType: boolean) =>
  tileType ? TileType.Ground : TileType.Water;

const mapPosition = (raw: Array<number>) => p(raw[1], raw[0]);

const mapState = (engineState: EngineState) =>
  state(
    engineState.energy,
    engineState.players.map(mapPlayer),
    engineState.lighthouses.map(mapLighthouse),
  );

const mapRound = (engineRound: EngineRound, index: number) => {
  const roundName = `Round ${index + 1}`;
  const roundState = mapState(engineRound.setup);
  const initialState = JSON.parse(JSON.stringify(roundState)) as State;
  const turns = engineRound.turns.map((turn) => mapTurn(turn, initialState));
  return round(roundName, roundState, turns);
};

const mapTurn = (engineTurn: EngineTurn, initialState: State) => {
  const player = mapPlayer(engineTurn.player);
  const turnName = `${player.name} turn`;
  const turnPlayers = initialState.players.map((roundPlayer) =>
    roundPlayer.id === player.id ? player : roundPlayer,
  );
  const turnState = state(
    initialState.energy,
    turnPlayers,
    initialState.lighthouses,
  );
  Object.assign(initialState, turnState);
  return turn(turnName, turnState);
};

const mapPlayer = (enginePlayer: EnginePlayer) =>
  player(
    enginePlayer.id,
    enginePlayer.name,
    enginePlayer.energy,
    enginePlayer.score,
    enginePlayer.keys,
    colorFor(enginePlayer.id - 1),
    mapPosition(enginePlayer.position),
  );

const mapLighthouse = (engineLighthouse: EngineLighthouse) =>
  lighthouse(
    engineLighthouse.id,
    engineLighthouse.energy,
    engineLighthouse.ownerId,
    engineLighthouse.connections,
    mapPosition(engineLighthouse.position),
  );
