import { 
  Game, 
  player, 
  lighthouse, 
  round,
  initialRoundStatus,
  boardStatus,
  playerStatus,
  playerScore,
  lighthouseStatus,
  turn,
  c
} from '@/code/domain.js'

const palette = [
  0xff0000,
  0x580aff,
  0xdeff0a,
  0xbe0aff,
  0x0aefff,
  0xa1ff0a,
  0xff8700,
  0x147df5,
  0x0aff99,
  0xffd300,
]

export const two_players = () => {
  const map = [
    ['w', 'w', 'w', 'w', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'w', 'w', 'w', 'w'],
  ]

  const player1 = player(1, 'Player 1', palette[0], c(1, 1))
  const player2 = player(2, 'Player 2', palette[1], c(1, 2))
  const players = [player1, player2]

  const lighthouse1 = lighthouse(1, c(2, 2))
  const lighthouses = [lighthouse1]

  const rounds = [
    round(
      1,
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 0, 4, 4, 0],
          [0, 0, 5, 100, 0],
          [0, 4, 4, 50, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 4, []), playerStatus(player2.id, 4, [])],
        [playerScore(player1.id, 0), playerScore(player2.id, 0)],
        [lighthouseStatus(lighthouse1.id, null, 0, [])]
      ), [
        turn(playerStatus(player1.id, 4, []), c(2, 1)),
        turn(playerStatus(player2.id, 4, []), c(1, 3)),
      ],
    ),
    round(
      2, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 4, 0, 8, 0],
          [0, 4, 10, 100, 0],
          [0, 0, 8, 50, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 12, []), playerStatus(player2.id, 12, [])],
        [playerScore(player1.id, 0), playerScore(player2.id, 0)],
        [lighthouseStatus(lighthouse1.id, null, 0, [])]
      ), [
        turn(playerStatus(player1.id, 12, []), c(2, 2)),
        turn(playerStatus(player2.id, 12, []), c(2, 3)),
      ],
    ),
    round(
      3, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 8, 4, 12, 0],
          [0, 8, 0, 100, 0],
          [0, 4, 0, 50, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 27, [1]), playerStatus(player2.id, 24, [])],
        [playerScore(player1.id, 2), playerScore(player2.id, 0)],
        [lighthouseStatus(lighthouse1.id, null, 0, [])]
      ), [
        turn(playerStatus(player1.id, 0, [1]), c(2, 2), lighthouseStatus(lighthouse1.id, 1, 27, [])),
        turn(playerStatus(player2.id, 24, []), c(2, 2)),
      ],
    ),
    round(
      4, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 12, 8, 16, 0],
          [0, 12, 0, 100, 0],
          [0, 8, 4, 50, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 2, [1]), playerStatus(player2.id, 26, [1])],
        [playerScore(player1.id, 2), playerScore(player2.id, 2)],
        [lighthouseStatus(lighthouse1.id, 1, 17, [])]
      ), [
        turn(playerStatus(player1.id, 2, [1]), c(1, 2)),
        turn(playerStatus(player2.id, 0, [1]), c(2, 2), lighthouseStatus(lighthouse1.id, 2, 9, [])),
      ],
    ),
  ]

  return new Game(map, players, lighthouses, rounds)
}

export const three_players = () => {
  const map = [
    ['w', 'w', 'w', 'w', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'g', 'g', 'g', 'w'],
    ['w', 'w', 'w', 'w', 'w'],
  ]

  const player1 = player(1, 'Player 1', palette[0], c(1, 1))
  const player2 = player(2, 'Player 2', palette[1], c(1, 2))
  const player3 = player(3, 'Player 3', palette[2], c(1, 3))
  const players = [player1, player2, player3]

  const lighthouse1 = lighthouse(1, c(2, 2))
  const lighthouses = [lighthouse1]

  const rounds = [
    round(
      1, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
        [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
        [lighthouseStatus(lighthouse1.id, 0)]
      ), [
        turn(playerStatus(player1.id, 4, []), c(2, 2)),
        turn(playerStatus(player2.id, 4, []), c(2, 2)),
        turn(playerStatus(player3.id, 4, []), c(2, 2)),
      ],
    ),
    round(
      2, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
        [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
        [lighthouseStatus(lighthouse1.id, 0)]
      ), [
        turn(playerStatus(player1.id, 12, []), c(3, 3)),
        turn(playerStatus(player2.id, 12, []), c(3, 2)),
        turn(playerStatus(player3.id, 12, []), c(3, 1)),
      ],
    ),
    round(
      3, 
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
        ]),
        [playerStatus(player1.id, 0, []), playerStatus(player2.id, 0, []), playerStatus(player3.id, 0, [])],
        [playerScore(player1.id, 0), playerScore(player2.id, 0), playerScore(player3.id, 0)],
        [lighthouseStatus(lighthouse1.id, 0)]
      ), [
        turn(playerStatus(player1.id, 12, []), c(3, 2)),
        turn(playerStatus(player2.id, 12, []), c(3, 2)),
        turn(playerStatus(player3.id, 12, []), c(3, 2)),
      ],
    )
  ]

  return new Game(map, players, lighthouses, rounds)
}

export const colors_demo = () => {
  const map = [
    ['g', 'g', 'g', 'g'],
    ['g', 'g', 'g', 'g'],
    ['g', 'g', 'g', 'g'],
  ]

  const player1 = player(1, 'Player 1', palette[0], c(0, 0))
  const player2 = player(2, 'Player 2', palette[1], c(0, 1))
  const player3 = player(3, 'Player 3', palette[2], c(0, 2))
  const player4 = player(4, 'Player 4', palette[3], c(0, 3))
  const player5 = player(5, 'Player 5', palette[4], c(1, 0))
  const player6 = player(6, 'Player 6', palette[5], c(1, 1))
  const player7 = player(7, 'Player 7', palette[6], c(1, 2))
  const player8 = player(8, 'Player 8', palette[7], c(1, 3))
  const player9 = player(9, 'Player 9', palette[8], c(2, 0))
  const player10 = player(10, 'Player 10', palette[9], c(2, 1))

  const rounds = [
    round(
      1,
      initialRoundStatus(
        boardStatus([
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ]),
        [
          playerStatus(player1.id, 0, []), 
          playerStatus(player2.id, 0, []),
          playerStatus(player3.id, 0, []),
          playerStatus(player4.id, 0, []),
          playerStatus(player5.id, 0, []),
          playerStatus(player6.id, 0, []),
          playerStatus(player7.id, 0, []),
          playerStatus(player8.id, 0, []),
          playerStatus(player9.id, 0, []),
          playerStatus(player10.id, 0, []),
        ],
        [
          playerScore(player1.id, 0), 
          playerScore(player2.id, 0),
          playerScore(player3.id, 0),
          playerScore(player4.id, 0),
          playerScore(player5.id, 0),
          playerScore(player6.id, 0),
          playerScore(player7.id, 0),
          playerScore(player8.id, 0),
          playerScore(player9.id, 0),
          playerScore(player10.id, 0),
        ],
        []
      ), 
      []
    ),
  ]

  return new Game(map, [
    player1, player2, player3, player4, player5, player6, player7, player8, player9, player10
  ], [], rounds)
}

