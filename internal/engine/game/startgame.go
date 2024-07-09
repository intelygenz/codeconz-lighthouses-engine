package game

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"time"

	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/xy"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()

	for i := 0; i < e.turns; i++ {
		time.Sleep(4 * time.Second)
		e.gameMap.CalcIslandEnergy()
		e.gameMap.CalcLighthouseEnergy()
		for _, p := range e.players {
			e.gameMap.CalcPlayerEnergy(e.GetPlayers(), p)

			na, err := p.RequestNewTurn(player.Turn{
				Position:    p.Position,
				Score:       p.Score,
				Energy:      p.Energy,
				View:        e.gameMap.GetPlayerView(p),
				Lighthouses: e.gameMap.GetLightHouses(),
			})
			if err != nil {
				// handle error
				fmt.Printf("Player %d has error %v", p.ID, err)
				break
			}

			// TODO: apply the action to change the state of the game
			cmd := exec.Command("clear") //Linux example, its tested
			cmd.Stdout = os.Stdout
			cmd.Run()
			err = e.movePlayer(p, na)
			if err != nil {
				fmt.Printf("Player %d has error %v", p.ID, err)
			}
		}
		// e.CalcPlayersScores()
		// TODO: calc players scores
		e.gameMap.PrettyPrintMap(e.players)
	}
}

func (e *Game) movePlayer(p *player.Player, action *player.Action) error {
	fmt.Printf("Player %d action %v\n", p.ID, action)

	switch action.Action {
	case player.ActionMove:
		return e.moveToPosition(p, action)
	case player.ActionAttack:
		return e.attackPosition(p, action)
	case player.ActionConnect:
		return e.connectLighthouses()
	case player.ActionPass:
		fmt.Printf("Player %d pass\n", p.ID)
	}

	return nil
}

func (e *Game) moveToPosition(p *player.Player, action *player.Action) error {
	fmt.Printf("Player %d moving from %v to %v\n", p.ID, p.Position, action.Destination)
	if xy.Distance(p.Position, action.Destination) >= 2 {
		return fmt.Errorf("player %d can't move to %v", p.ID, action.Destination)
	}

	if e.gameMap.CanMoveTo(action.Destination) == false {
		return fmt.Errorf("player %d can't move to %v", p.ID, action.Destination)
	}

	fmt.Printf("Player %d moving from %v to %v\n", p.ID, p.Position, action.Destination)
	p.Position = action.Destination
	return nil
}

func (e *Game) attackPosition(p *player.Player, action *player.Action) error {
	fmt.Printf("Player %d attack\n", p.ID)

	if action.Energy > p.Energy {
		return fmt.Errorf("player %d has no energy to attack", p.ID)
	}

	for _, l := range e.gameMap.GetLightHouses() {
		if l.Position.Equal(geom.XY, action.Destination) {
			if l.Owner == p.ID {
				l.Energy += action.Energy
			}

			lighthouseEnergy := l.Energy - action.Energy

			if lighthouseEnergy == 0 {
				l.Energy = 0
				l.Owner = -1
			}

			if lighthouseEnergy < 0 {
				l.Energy = int(math.Abs(float64(lighthouseEnergy)))
				l.Owner = p.ID
			}
			break
		}
	}
	return nil
}

func (e *Game) connectLighthouses() error {
	// TODO: implement connect action and calculate connections
	return nil
}
