package game

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"strconv"
	"time"
)

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()
	/*
		viewExample := &coms.MapRow{
			Row: []int32{0, 0, 0, 0, 0},
		}
		lighthouseExample := &coms.Lighthouse{
			Position: &coms.Position{
				X: 1,
				Y: 1,
			},
			Energy: 100,
		}

		newTurnExample := &coms.NewTurn{
			Position: &coms.Position{
				X: 1,
				Y: 1,
			},
			Score:       10,
			Energy:      100,
			View:        []*coms.MapRow{viewExample},
			Lighthouses: []*coms.Lighthouse{lighthouseExample},
		}*/

	for i := 0; i < e.turns; i++ {
		// TODO: calc new turn state
		// e.gameMap.CalcEnergy()
		for _, p := range e.players {
			gc := &handler.GameClient{}
			comPlayer := &coms.NewPlayer{
				Name:          strconv.Itoa(p.ID), //TODO: replace name with ID
				ServerAddress: p.ServerAddress,
			}
			// TODO: calc the new turn with the last state of the game
			na := gc.RequestTurn(comPlayer, nil /*e.gameMap.getPlayerView(p)*/)
			_ = na
			// TODO: apply the action to change the state of the game
			e.movePlayer(p, na)
		}
		// e.CalcPlayersScores()
		// TODO: calc players scores
	}
}

func (e *Game) movePlayer(p player.Player, na *coms.NewAction) {
	switch na.Action {
	case coms.Action_MOVE:

	case coms.Action_PASS:

	case coms.Action_ATTACK:

	case coms.Action_CONNECT:

	}
}
