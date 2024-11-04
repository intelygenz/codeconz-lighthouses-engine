import {
  game,
  setup,
  player,
  lighthouse,
  round,
  turn,
  p
} from '@/code/domain.js'

import raw_engine_game_1 from '@/assets/game-2024_10_07_15_27_54.json'
import raw_engine_game_2 from '@/assets/game-2024_10_10_08_49_43.json'
import raw_engine_game_3 from '@/assets/game-2024_10_10_11_41_32.json'
import raw_engine_game_4 from '@/assets/game-2024_10_16_15_34_24.json'
import raw_engine_game_5 from '@/assets/game-2024_10_18_09_51_35.json'

const palette = [
  0xff0000,
  0x580aff,
  0xdeff0a,
  0xbe0aff,
  0x0aefff,
  0x51ff0a,
  0xff8700,
  0x147df5,
  0x0aff99,
  0xffd300,
]

const parsePosition = (raw) => (p(raw[1], raw[0]))

const parseLighthouse = (raw) => (lighthouse(
  raw.id,
  raw.energy,
  raw.ownerId,
  raw.connections,
  parsePosition(raw.position),
))

const parsePlayer = (raw) => (player(
  raw.id,
  parsePosition(raw.position),
  raw.energy,
  raw.score,
  raw.keys,
  raw.name,
  palette[raw.id - 1],
))

const parseTurn = (raw) => (turn(
  parsePlayer(raw.player),
  raw.lighthouses.map(parseLighthouse),
))

const parseRound = (raw) => (round(
  setup(
    raw.setup.energy,
    raw.setup.players.map(parsePlayer),
    raw.setup.lighthouses.map(parseLighthouse),
  ),
  raw.turns.map(parseTurn),
))


const parse = (raw) => (game(
  raw.topology.map(row => row.map(col => col ? 'g' : 'w')),
  setup(
    raw.setup.energy,
    raw.setup.players.map(parsePlayer),
    raw.setup.lighthouses.map(parseLighthouse),
  ),
  raw.rounds.map(parseRound),
))

export const engine_game_1 = parse(raw_engine_game_1)
export const engine_game_2 = parse(raw_engine_game_2)
export const engine_game_3 = parse(raw_engine_game_3)
export const engine_game_4 = parse(raw_engine_game_4)
export const engine_game_5 = parse(raw_engine_game_5)

const map_1_tiles = [
  '                     ',
  '  gggg ggg  ggg    g ',
  ' g  gg  g  g gggg gg ',
  ' gg  gg g gg ggggg g ',
  ' g g  ggggg  ggggggg ',
  ' g ggg gggg gggggggg ',
  '  gggg gg       g gg ',
  ' ggggg g gggg  g  gg ',
  ' gg   g  gg   gggg   ',
  ' g   ggg  ggg  ggg   ',
  ' g g ggggg ggg gg g  ',
  '  ggg  gggg ggg g  g ',
  '  g    ggg   g ggg   ',
  ' g    gggg    ggg g  ',
  '                     ',
]

export const map_1 = game(
  map_1_tiles.map(row => row.split('').map(col => col === 'g' ? 'g' : 'w')),
  setup([], [
  ], [
  ]),
  []
)


export const simple_demo = () => game([
    ['w', 'w', 'w', 'w', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'w', 'w', 'w', 'w'],
  ],
  setup([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 100, 0],
    [0, 0, 0, 50, 0],
    [0, 0, 0, 0, 0],
  ], [
    player(1, p(1, 1), 0, 0, [], 'Player 1', palette[0]),
    player(2, p(1, 2), 0, 0, [], 'Player 2', palette[1])
  ], [
    lighthouse(1, 0, null, [], p(2, 2))
  ]),
  [
    round(
      setup([
        [0, 0, 0, 0, 0],
        [0, 0, 4, 4, 0],
        [0, 0, 5, 100, 0],
        [0, 4, 4, 50, 0],
        [0, 0, 0, 0, 0],
      ], [
        player(1, p(1, 1), 4, 0, []),
        player(2, p(1, 2), 4, 0, [])
      ], [
        lighthouse(1, 0, null, [])
      ]),
      [
        turn(player(1, p(2, 1), 4, 0, []), [
          lighthouse(1, 0, null, [])
        ]),
        turn(player(2, p(1, 3), 4, 0, []), [
          lighthouse(1, 0, null, [])
        ])
      ]),
    round(
      setup([
        [0, 0, 0, 0, 0],
        [0, 4, 0, 8, 0],
        [0, 4, 10, 100, 0],
        [0, 0, 8, 50, 0],
        [0, 0, 0, 0, 0],
      ], [
        player(1, p(2, 1), 12, 0, []),
        player(2, p(1, 3), 12, 0, [])
      ], [
        lighthouse(1, 0, null, [])
      ]),
      [
        turn(player(1, p(2, 2), 12, 0, []), [
          lighthouse(1, 0, null, [])
        ]),
        turn(player(2, p(2, 3), 12, 0, []), [
          lighthouse(1, 0, null, [])
        ])
      ]),
    round(
      setup([
          [0, 0, 0, 0, 0],
          [0, 8, 4, 12, 0],
          [0, 8, 0, 100, 0],
          [0, 4, 0, 50, 0],
          [0, 0, 0, 0, 0],
      ], [
        player(1, p(2, 2), 27, 0, [1]),
        player(2, p(2, 3), 24, 0, [])
      ], [
        lighthouse(1, 0, null, [])
      ]),
      [
        turn(player(1, p(2, 2), 0, 0, []), [
          lighthouse(1, 27, 1, [])
        ]),
        turn(player(2, p(2, 2), 24, 0, []), [
          lighthouse(1, 27, 1, [])
        ])
      ]),
    round(
      setup([
          [0, 0, 0, 0, 0],
          [0, 12, 8, 16, 0],
          [0, 12, 0, 100, 0],
          [0, 8, 4, 50, 0],
          [0, 0, 0, 0, 0],
      ], [
        player(1, p(2, 2), 2, 0, [1]),
        player(2, p(2, 2), 26, 0, [])
      ], [
        lighthouse(1, 17, 1, [])
      ]),
      [
        turn(player(1, p(1, 2), 2, 0, []), [
          lighthouse(1, 17, 1, [])
        ]),
        turn(player(2, p(2, 2), 0, 0, []), [
          lighthouse(1, 9, 2, [])
        ])
      ]),
  ])

export const nine_players = () => game([
    ['w', 'w', 'w', 'w', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'w', 'g', 'w', 'w'],
    ['w', 'w', 'w', 'w', 'w'],
  ],
  setup([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ], [
    player(1, p(1, 1), 0, 0, [], 'Player', palette[0]),
    player(2, p(2, 1), 0, 0, [], 'Player', palette[1]),
    player(3, p(3, 1), 0, 0, [], 'Player', palette[2]),
    player(4, p(1, 2), 0, 0, [], 'Player', palette[3]),
    player(5, p(2, 2), 0, 0, [], 'Player', palette[4]),
    player(6, p(3, 2), 0, 0, [], 'Player', palette[5]),
    player(7, p(1, 3), 0, 0, [], 'Player', palette[6]),
    player(8, p(2, 3), 0, 0, [], 'Player', palette[7]),
    player(9, p(3, 3), 0, 0, [], 'Player', palette[8]),
    player(10, p(2, 4), 0, 0, [], 'Player', palette[9]),
  ], [
    lighthouse(1, 0, 1, [], p(1, 1)),
    lighthouse(2, 0, 2, [], p(2, 1)),
    lighthouse(3, 0, 3, [], p(3, 1)),
    lighthouse(4, 0, 4, [], p(1, 2)),
    lighthouse(5, 0, 5, [], p(2, 2)),
    lighthouse(6, 0, 6, [], p(3, 2)),
    lighthouse(7, 0, 7, [], p(1, 3)),
    lighthouse(8, 0, 8, [], p(2, 3)),
    lighthouse(9, 0, 9, [], p(3, 3)),
    lighthouse(10, 0, 10, [], p(2, 4)),
  ]),
  [
    round(
      setup([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
      ], [
        player(1, p(2, 2), 0, 0, [], 'Player', palette[0]),
        player(2, p(2, 2), 0, 0, [], 'Player', palette[1]),
        player(3, p(2, 2), 0, 0, [], 'Player', palette[2]),
        player(4, p(2, 2), 0, 0, [], 'Player', palette[3]),
        player(5, p(2, 2), 0, 0, [], 'Player', palette[4]),
        player(6, p(2, 2), 0, 0, [], 'Player', palette[5]),
        player(7, p(2, 2), 0, 0, [], 'Player', palette[6]),
        player(8, p(2, 2), 0, 0, [], 'Player', palette[7]),
        player(9, p(2, 2), 0, 0, [], 'Player', palette[8]),
        player(10, p(2, 2), 0, 0, [], 'Player', palette[9]),
      ], [
      ]),
      [
      ]),
  ])

// export const three_players = () => {
//   const map = [
//     ['w', 'w', 'w', 'w', 'w'],
//     ['w', 'g', 'g', 'g', 'w'],
//     ['w', 'g', 'g', 'g', 'w'],
//     ['w', 'g', 'g', 'g', 'w'],
//     ['w', 'w', 'w', 'w', 'w'],
//   ]
//
//   const player1 = player(1, 'Player 1', palette[0], c(1, 1))
//   const player2 = player(2, 'Player 2', palette[1], c(1, 2))
//   const player3 = player(3, 'Player 3', palette[2], c(1, 3))
//   const players = [player1, player2, player3]
//
//   const lighthouse1 = lighthouse(1, c(2, 2))
//   const lighthouses = [lighthouse1]
//
//   const rounds = [
//     round(
//       1,
//       roundStatus(
//         boardStatus([
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//         ]),
//         [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
//         [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
//         [lighthouseStatus(lighthouse1.id, 0)]
//       ), [
//         turn(playerStatus(player1.id, 4, []), c(2, 2)),
//         turn(playerStatus(player2.id, 4, []), c(2, 2)),
//         turn(playerStatus(player3.id, 4, []), c(2, 2)),
//       ],
//     ),
//     round(
//       2,
//       roundStatus(
//         boardStatus([
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//         ]),
//         [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
//         [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
//         [lighthouseStatus(lighthouse1.id, 0)]
//       ), [
//         turn(playerStatus(player1.id, 12, []), c(3, 3)),
//         turn(playerStatus(player2.id, 12, []), c(3, 2)),
//         turn(playerStatus(player3.id, 12, []), c(3, 1)),
//       ],
//     ),
//     round(
//       3,
//       roundStatus(
//         boardStatus([
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0],
//         ]),
//         [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
//         [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
//         [lighthouseStatus(lighthouse1.id, 0)]
//       ), [
//         turn(playerStatus(player1.id, 12, []), c(3, 2)),
//         turn(playerStatus(player2.id, 12, []), c(3, 2)),
//         turn(playerStatus(player3.id, 12, []), c(3, 2)),
//       ],
//     )
//   ]
//
//   return new Game(map, players, lighthouses, rounds)
// }
//
// export const colors_demo = () => {
//   const map = [
//     ['g', 'g', 'g', 'g'],
//     ['g', 'g', 'g', 'g'],
//     ['g', 'g', 'g', 'g'],
//   ]
//
//   const player1 = player(1, 'Player 1', palette[0], c(0, 0))
//   const player2 = player(2, 'Player 2', palette[1], c(0, 1))
//   const player3 = player(3, 'Player 3', palette[2], c(0, 2))
//   const player4 = player(4, 'Player 4', palette[3], c(0, 3))
//   const player5 = player(5, 'Player 5', palette[4], c(1, 0))
//   const player6 = player(6, 'Player 6', palette[5], c(1, 1))
//   const player7 = player(7, 'Player 7', palette[6], c(1, 2))
//   const player8 = player(8, 'Player 8', palette[7], c(1, 3))
//   const player9 = player(9, 'Player 9', palette[8], c(2, 0))
//   const player10 = player(10, 'Player 10', palette[9], c(2, 1))
//
//   const rounds = [
//     round(
//       1,
//       roundStatus(
//         boardStatus([
//           [0, 0, 0, 0],
//           [0, 0, 0, 0],
//           [0, 0, 0, 0],
//         ]),
//         [
//           playerStatus(player1.id, 0, []),
//           playerStatus(player2.id, 0, []),
//           playerStatus(player3.id, 0, []),
//           playerStatus(player4.id, 0, []),
//           playerStatus(player5.id, 0, []),
//           playerStatus(player6.id, 0, []),
//           playerStatus(player7.id, 0, []),
//           playerStatus(player8.id, 0, []),
//           playerStatus(player9.id, 0, []),
//           playerStatus(player10.id, 0, []),
//         ],
//         [
//           playerScore(player1.id, 0),
//           playerScore(player2.id, 0),
//           playerScore(player3.id, 0),
//           playerScore(player4.id, 0),
//           playerScore(player5.id, 0),
//           playerScore(player6.id, 0),
//           playerScore(player7.id, 0),
//           playerScore(player8.id, 0),
//           playerScore(player9.id, 0),
//           playerScore(player10.id, 0),
//         ],
//         []
//       ),
//       []
//     ),
//   ]
//
//   return new Game(map, [
//     player1, player2, player3, player4, player5, player6, player7, player8, player9, player10
//   ], [], rounds)
// }
//
