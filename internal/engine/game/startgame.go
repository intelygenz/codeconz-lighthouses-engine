package game

import (
	"fmt"
	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/xy"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()

	for i := 0; i < e.turns; i++ {
		e.gameMap.CalcEnergy()
		for _, p := range e.players {
			// TODO: calc the energy of the player

			// TODO: calc the new turn with the last state of the game
			na, err := p.RequestNewTurn(player.Turn{
				Position:    p.Position,
				Score:       int32(p.Score),
				Energy:      int32(p.Energy),
				View:        [][]int32{{0, 1, 0, 0, 1}, {0, 1, 0, 0, 1}, {0, 1, 0, 0, 1}, {0, 1, 0, 0, 1}}, // TODO generate turn view
				Lighthouses: []geom.Coord{},
			})
			if err != nil {
				// handle error
				fmt.Printf("Player %d has error %v", p.ID, err)
				break
			}
			_ = na
			// TODO: apply the action to change the state of the game
			err = e.movePlayer(p, na)
		}
		// e.CalcPlayersScores()
		// TODO: calc players scores
		e.gameMap.PrettyPrintMap(e.players)
	}
	//for i := 0; i < e.turns; i++ {
	//	// TODO: calc new turn state
	//	e.gameMap.CalcEnergy()
	//	for _, p := range e.players {
	//		///*gc := &handler.GameClient{}
	//		//comPlayer := &coms.NewPlayer{
	//		//	Name:          strconv.Itoa(p.ID), //TODO: replace name with ID
	//		//	ServerAddress: p.ServerAddress,
	//		//}
	//		//// TODO: calc the new turn with the last state of the game
	//		//// na := gc.RequestTurn(comPlayer, nil /*e.gameMap.getPlayerView(p)*/)
	//		//// TODO: apply the action to change the state of the game
	//		//// e.movePlayer(p, na)*/
	//	}
	//	// e.CalcPlayersScores()
	//	// TODO: calc players scores
	//}
}

func (e *Game) movePlayer(p *player.Player, action *player.Action) error {
	fmt.Printf("Player %d action %v\n", p.ID, action)

	switch action.Action {
	case player.ActionMove:
		if xy.Distance(p.Position, action.Destination) >= 2 {
			return fmt.Errorf("player %d can't move to %v", p.ID, action.Destination)
		}

		fmt.Printf("Player %d moving from %v to %v\n", p.ID, p.Position, action.Destination)
		p.Position = action.Destination
	case player.ActionAttack:
	case player.ActionConnect:
	case player.ActionPass:

	}

	return nil
}
