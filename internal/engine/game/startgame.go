package game

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/spf13/viper"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/state"
)

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()
	// generate initial game state
	gStatus := state.NewGameStatus(e.gameMap.GetIslandEnergy(), e.GetPlayers(), e.gameMap.GetLightHouses())
	gState := state.GameState{
		Topology: e.gameMap.GetPlayableMap(),
		Setup:    gStatus,
	}
	e.state.SetGameState(gState)

	for i := 0; i < e.turns; i++ {
		roundId := i + 1
		if viper.GetBool("game.verbosity") {
			cmd := exec.Command("clear") //Linux example, its tested
			cmd.Stdout = os.Stdout
			cmd.Run()
		}
		timeBetweenRounds := viper.GetDuration("game.time_between_rounds")
		if timeBetweenRounds > 0 {
			time.Sleep(timeBetweenRounds)
		}
		e.gameMap.CalcIslandEnergy()
		e.gameMap.CalcLighthouseEnergy()

		// generate initial round state
		round := state.NewGameStatus(e.gameMap.GetIslandEnergy(), e.GetPlayers(), e.gameMap.GetLightHouses())
		e.state.SetNewRound(roundId, round)

		// give energy to all players before turns starts
		for _, p := range e.players {
			e.gameMap.CalcPlayerEnergy(e.GetPlayers(), p)
		}

		for _, p := range e.players {
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

			// generate turn state and set into game state
			turn := state.NewTurn(p, e.gameMap.GetLightHouses())
			e.state.AddPlayerTurn(roundId, turn)
		}

		e.CalcPlayersScores()

		if viper.GetBool("game.verbosity") {
			e.gameMap.PrettyPrintMap(e.players)
		}
	}

	// generate final game state after all turns in last round
	gfStatus := state.NewGameStatus(e.gameMap.GetIslandEnergy(), e.GetPlayers(), e.gameMap.GetLightHouses())
	e.state.SetFinalGameState(gfStatus)

	// dump to file the final state of the game in json format
	if err := e.state.DumpToFileFinalStateInJson(); err != nil {
		fmt.Printf("State to json could not be generated: %v\n", err)
	}
}
