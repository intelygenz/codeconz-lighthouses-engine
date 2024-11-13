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

import eg_2p from "@/games/game-2024_11_11_21_29_31.json";
import eg_9p from "@/games/game-2024_11_11_21_38_03.json";
import eg_demo from "@/games/game-2024_11_12_12_22_16.json";
import eg_demo_2 from "@/games/game-2024_11_12_12_30_33.json";
import eg_demo_3 from "@/games/game-2024_11_12_12_38_13.json";

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

export const g2p = parse(eg_2p);
export const g9p = parse(eg_9p);
export const demo = parse(eg_demo);
export const demo2 = parse(eg_demo_2);
export const demo3 = parse(eg_demo_3);
