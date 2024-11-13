export interface Game {
  board: Board;
  state: State;
  rounds: Array<Round>;
  finalState: State;
}

export const game = (
  board: Board,
  state: State,
  rounds: Array<Round>,
  finalState: State,
): Game => ({ board, state, rounds, finalState });

export interface Board extends Array<Array<Tile>> {}

export interface Tile extends Position {
  type: TileType;
  energy: number;
}

export const tile = (
  type: TileType,
  energy: number,
  position: Position,
): Tile => ({ type, energy, x: position.x, y: position.y });

export enum TileType {
  Ground = "g",
  Water = "w",
}

export interface State {
  energy: Array<Array<number>>;
  players: Array<Player>;
  lighthouses: Array<Lighthouse>;
}

export const state = (
  energy: Array<Array<number>>,
  players: Array<Player>,
  lighthouses: Array<Lighthouse>,
): State => ({ energy, players, lighthouses });

export interface Player extends Position {
  id: number;
  name: string;
  energy: number;
  score: number;
  keys: Array<number>;
  color: number;
}

export const player = (
  id: number,
  name: string,
  energy: number,
  score: number,
  keys: Array<number>,
  color: number,
  position: Position,
): Player => ({
  id,
  name,
  energy,
  score,
  keys,
  color,
  x: position.x,
  y: position.y,
});

export interface Lighthouse extends Position {
  id: number;
  energy: number;
  ownerId: number;
  links: Array<number>;
}

export const lighthouse = (
  id: number,
  energy: number,
  ownerId: number,
  links: Array<number>,
  position: Position,
): Lighthouse => ({ id, energy, ownerId, links, x: position.x, y: position.y });

export interface Round {
  name: string;
  state: State;
  turns: Array<Turn>;
}

export const round = (
  name: string,
  state: State,
  turns: Array<Turn>,
): Round => ({ name, state, turns });

export interface Turn {
  name: string;
  state: State;
}

export const turn = (name: string, state: State): Turn => ({ name, state });

export interface Position {
  x: number;
  y: number;
}

export const p = (x: number, y: number): Position => ({ x, y });
