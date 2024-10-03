package state

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

type State interface {
	SetGameState(GameState)
	SetNewRound(int, GameStatus)
	AddPlayerTurn(int, Turn)
	DumpToFileFinalStateInJson() error
}

type GameStatus struct {
	Energy     [][]int                  `json:"energy"` // TODO: see how to get easily energy
	Players    []*player.Player         `json:"players"`
	Lighthouse []*lighthouse.Lighthouse `json:"lighthouses"`
}

type Turn struct {
	Player      player.Player            `json:"player"`
	Lighthouses []*lighthouse.Lighthouse `json:"lighthouses"`
}
type Round struct {
	Id    int        `json:"-"`
	Setup GameStatus `json:"setup"`
	Turns []Turn     `json:"turns"`
}

type GameState struct {
	Topology [][]bool   `json:"topology"` // TODO: game.board.cells
	Setup    GameStatus `json:"setup"`
	Rounds   []Round    `json:"rounds"`
}
