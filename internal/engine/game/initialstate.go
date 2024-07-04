package game

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/twpayne/go-geom"
)

type PlayerInitialState struct {
	PlayerID    int
	PlayerCount int
	Position    geom.Coord
	Map         [][]bool
	Lighthouses []lighthouse.Lighthouse
}

func (e *Game) CreateInitialState(p player.Player) *PlayerInitialState {
	npst := &PlayerInitialState{
		PlayerID:    p.ID,
		PlayerCount: len(e.players),
		Position:    p.Position,
		Map:         e.gameMap.GetPlayableMap(),
		Lighthouses: e.gameMap.GetLightHouses(),
	}
	return npst
}
