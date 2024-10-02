package game

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"time"

	"github.com/spf13/viper"
	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/xy"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()

	for i := 0; i < e.turns; i++ {
		if viper.GetBool("game.verbosity") {
			cmd := exec.Command("clear") //Linux example, its tested
			cmd.Stdout = os.Stdout
			cmd.Run()
		}
		time.Sleep(4 * time.Second) // TODO remove sleep for real game
		e.gameMap.CalcIslandEnergy()
		e.gameMap.CalcLighthouseEnergy()
		for _, p := range e.players {
			e.gameMap.CalcPlayerEnergy(e.GetPlayers(), p)

			na, err := p.RequestNewTurn(player.Turn{
				Position:           p.Position,
				Score:              p.Score,
				Energy:             p.Energy,
				View:               e.gameMap.GetPlayerView(p),
				Lighthouses:        e.gameMap.GetLightHouses(),
				UserLighthouseKeys: p.LighthouseKeys,
			})
			if err != nil {
				// handle error
				fmt.Printf("Player %d has error %v\n", p.ID, err)
				break
			}

			err = e.execPlayerAction(p, na)
			if err != nil {
				fmt.Printf("Player %d has error %v\n", p.ID, err)
			}
			fmt.Println("*************************************************")
		}

		e.CalcPlayersScores()

		if viper.GetBool("game.verbosity") {
			e.gameMap.PrettyPrintMap(e.players)
		}
	}
}

func (e *Game) execPlayerAction(p *player.Player, action *player.Action) error {
	fmt.Printf("Player %d action %v\n", p.ID, action)

	switch action.Action {
	case player.ActionMove:
		return e.moveToPosition(p, action)
	case player.ActionAttack:
		return e.attackPosition(p, action)
	case player.ActionConnect:
		return e.connectLighthouses(p, action) //TODO: add connections logic
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

	for _, l := range e.gameMap.GetLightHouses() {
		if l.Position.Equal(geom.XY, action.Destination) {
			fmt.Printf("Player %d obtained lighthouse key for lighthouse %v\n", p.ID, l.Position)
			p.LighthouseKeys = append(p.LighthouseKeys, l)
			break
		}
	}

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

			p.Energy -= action.Energy
			break
		}
	}
	return nil
}

func (e *Game) connectLighthouses(p *player.Player, action *player.Action) error {
	curLighthousePos := p.Position
	destLighthousePos := action.Destination

	if curLighthousePos.Equal(geom.XY, destLighthousePos) {
		return fmt.Errorf("player %d can't connect to the same lighthouse", p.ID)
	}

	curLighthouse, err := e.gameMap.GetLightHouse(curLighthousePos)
	if err != nil {
		return err
	}

	destLighthouse, err := e.gameMap.GetLightHouse(destLighthousePos)
	if err != nil {
		return err
	}

	if curLighthouse.Owner != p.ID {
		return fmt.Errorf("player %d does not own lighthouse %v", p.ID, curLighthousePos)
	}

	if destLighthouse.Owner != p.ID {
		return fmt.Errorf("player %d does not own lighthouse %v", p.ID, destLighthousePos)
	}

	for _, l := range e.gameMap.GetLightHouses() {
		if xy.IsPointWithinLineBounds(l.Position, curLighthousePos, destLighthousePos) {
			return fmt.Errorf("connection cannot intersect a lighthouse")
		}

		for _, c := range l.Connections {
			if xy.DoLinesOverlap(l.Position, c.Position, curLighthousePos, destLighthousePos) {
				return fmt.Errorf("connection cannot intersect another connection")
			}
		}
	}

	err = curLighthouse.Connect(destLighthouse)
	if err != nil {
		return err
	}

	p.RemoveLighthouseKey(*destLighthouse)

	return nil
}
