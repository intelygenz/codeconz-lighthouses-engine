package game

import (
	"fmt"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/twpayne/go-geom"
)

type PlayerInitialState struct {
	PlayerID    int
	PlayerCount int
	Position    geom.Coord
	Map         [][]bool
	Lighthouses []*lighthouse.Lighthouse
}

func (e *Game) CreateInitialState(p *player.Player) *PlayerInitialState {
	fmt.Printf("Creating initial state for player %v", p)
	npst := &PlayerInitialState{
		PlayerID:    p.ID,
		PlayerCount: len(e.players),
		Position:    p.Position,
		Map:         e.gameMap.GetPlayableMap(),
		Lighthouses: e.gameMap.GetLightHouses(),
	}
	return npst
}
