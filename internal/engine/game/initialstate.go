package game

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"github.com/twpayne/go-geom"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type PlayerInitialState struct {
	PlayerID    int
	PlayerCount int
	Position    geom.Coord
	Map         [][]bool
	Lighthouses []*lighthouse.Lighthouse
}

func (e *Game) CreateInitialState(p *player.Player) *PlayerInitialState {
	fmt.Printf("Creating initial state for player %v\n", p)
	npst := &PlayerInitialState{
		PlayerID:    p.ID,
		PlayerCount: len(e.players),
		Position:    p.Position,
		Map:         e.gameMap.GetPlayableMap(),
		Lighthouses: e.gameMap.GetLightHouses(),
	}
	return npst
}

func (e *Game) SendInitialState() {
	for _, p := range e.GetPlayers() {
		// generate initial state
		pInitSt := e.CreateInitialState(p)

		mapper := Mapper{}
		comPlayerInitialState := mapper.MapPlayerInitialStateToComPlayerInitialState(pInitSt)

		if viper.GetBool("verbosity") {
			b, err := json.Marshal(comPlayerInitialState)
			if err != nil {
				panic(err)
			}
			fmt.Println(string(b))
		}

		// set grpc client
		grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
		grpcClient, err := grpc.NewClient(p.ServerAddress, grpcOpt)
		if err != nil {
			panic(err)
		}

		// Add 5 seconds timeout to the InitialState call
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// sending initial state to player
		fmt.Printf("Sending Initial State to Player %s to address %v\n", p.Name, p.ServerAddress)
		npjc := coms.NewGameServiceClient(grpcClient)
		playerReady, err := npjc.InitialState(ctx, comPlayerInitialState)

		if err != nil {
			panic(err)
		}
		fmt.Printf("SendInitialState complete with status %v\n", playerReady.Ready)
	}
}
