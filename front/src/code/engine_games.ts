import {
  game,
  setup,
  player,
  lighthouse,
  round,
  turn,
  p,
} from "@/code/domain.js";
import { colorFor } from "@/code/palette";

import engine_game_1 from "@/games/game-2024_10_07_15_27_54.json";
import engine_game_2 from "@/games/game-2024_10_10_08_49_43.json";
import engine_game_3 from "@/games/game-2024_10_10_11_41_32.json";
import engine_game_4 from "@/games/game-2024_10_16_15_34_24.json";
import engine_game_5 from "@/games/game-2024_10_18_09_51_35.json";

const parsePosition = (raw) => p(raw[1], raw[0]);

const parseLighthouse = (raw) =>
  lighthouse(
    raw.id,
    raw.energy,
    raw.ownerId,
    raw.connections,
    parsePosition(raw.position),
  );

const parsePlayer = (raw) =>
  player(
    raw.id,
    parsePosition(raw.position),
    raw.energy,
    raw.score,
    raw.keys,
    raw.name,
    colorFor(raw.id - 1),
  );

const parseTurn = (raw) =>
  turn(parsePlayer(raw.player), raw.lighthouses.map(parseLighthouse));

const parseRound = (raw, index) =>
  round(
    setup(
      raw.setup.energy,
      raw.setup.players.map(parsePlayer),
      raw.setup.lighthouses.map(parseLighthouse),
    ),
    raw.turns.map(parseTurn),
    index,
  );

const parse = (raw) =>
  game(
    raw.topology.map((row) => row.map((col) => (col ? "g" : "w"))),
    setup(
      raw.setup.energy,
      raw.setup.players.map(parsePlayer),
      raw.setup.lighthouses.map(parseLighthouse),
    ),
    raw.rounds.map(parseRound),
  );

export const game_1 = parse(engine_game_1);
export const game_2 = parse(engine_game_2);
export const game_3 = parse(engine_game_3);
export const game_4 = parse(engine_game_4);
export const game_5 = parse(engine_game_5);
