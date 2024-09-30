package game

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/spf13/viper"

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

			// send message to each Player with the info
			na, err := p.RequestNewTurn(player.Turn{
				Position:    p.Position,
				Score:       p.Score,
				Energy:      p.Energy,
				View:        e.gameMap.GetPlayerView(p),
				Lighthouses: e.gameMap.GetLightHouses(),
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
