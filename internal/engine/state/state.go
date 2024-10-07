package state

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

type State interface {
	SetGameState(GameState)
	SetFinalGameState(*GameStatus)
	SetNewRound(int, *GameStatus)
	AddPlayerTurn(int, *Turn)
	DumpToFileFinalStateInJson() error
}

type GameStatus struct {
	Energy      [][]int                 `json:"energy"`
	Players     []player.Player         `json:"players"`
	Lighthouses []lighthouse.Lighthouse `json:"lighthouses"`
}

func NewGameStatus(energy [][]int, players []*player.Player, lighthouses []*lighthouse.Lighthouse) *GameStatus {
	var pPlayers []player.Player
	for _, p := range players {
		p.GenerateLighthouseKeysIds()
		pPlayers = append(pPlayers, *p)
	}

	var pLighthouses []lighthouse.Lighthouse
	for _, l := range lighthouses {
		l.GenerateConnectionsId()
		pLighthouses = append(pLighthouses, *l)
	}
	return &GameStatus{
		Energy:      energy,
		Players:     pPlayers,
		Lighthouses: pLighthouses,
	}
}

type Turn struct {
	Player      player.Player           `json:"player"`
	Lighthouses []lighthouse.Lighthouse `json:"lighthouses"`
}

func NewTurn(player *player.Player, lighthouses []*lighthouse.Lighthouse) *Turn {
	player.GenerateLighthouseKeysIds()

	var pLighthouses []lighthouse.Lighthouse
	for _, l := range lighthouses {
		l.GenerateConnectionsId()
		pLighthouses = append(pLighthouses, *l)
	}

	return &Turn{
		Player:      *player,
		Lighthouses: pLighthouses,
	}

}

type Round struct {
	Id    int         `json:"-"`
	Setup *GameStatus `json:"setup"`
	Turns []*Turn     `json:"turns"`
}

type GameState struct {
	Topology    [][]bool    `json:"topology"`
	Setup       *GameStatus `json:"setup"`
	Rounds      []Round     `json:"rounds"`
	FinalStatus *GameStatus `json:"finalStatus"`
}
