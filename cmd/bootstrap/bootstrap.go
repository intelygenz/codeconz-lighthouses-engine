package bootstrap

import (
	"fmt"
	"net"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/game"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
)

type Bootstrap struct {
}

func NewBootstrap() *Bootstrap {
	return &Bootstrap{}
}

func (b *Bootstrap) Run() {
	b.initializeConfiguration()

	if viper.GetString("game.listen_address") == "" {
		panic("addr is required")
	}

	fmt.Println("Game server starting on", viper.GetString("game.listen_address"))

	lis, err := net.Listen("tcp", viper.GetString("game.listen_address"))
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(handler.UnaryLoggingInterceptor),
		grpc.StreamInterceptor(handler.StreamLoggingInterceptor),
	)

	ge := game.NewGame(viper.GetString("game.board_path"), viper.GetInt("game.turns"))
	gs := handler.NewGameServer(ge)

	coms.RegisterGameServiceServer(grpcServer, gs)

	go func() {
		<-time.After(viper.GetDuration("game.join_timeout"))
		grpcServer.Stop()
	}()

	if err := grpcServer.Serve(lis); err != nil {
		panic(err)
	}

	fmt.Println("players joining is closed.")

	if len(ge.GetPlayers()) == 0 {
		panic("No players joined, game did not start.")
	}

	fmt.Println("sending initial state to all players.")
	ge.SendInitialState()

	fmt.Println("Now the game starts!!!")
	ge.StartGame()
	fmt.Println("Game finished!!!")
}

func (b *Bootstrap) initializeConfiguration() {
	// TODO add ENV variable and configuration file support

	viper.SetDefault("game.listen_address", ":50051")
	viper.SetDefault("game.join_timeout", 5*time.Second)
	viper.SetDefault("game.turn_request_timeout", 100*time.Millisecond)
	viper.SetDefault("game.turns", 15)
	viper.SetDefault("game.board_path", "./maps/island_simple.txt")
	viper.SetDefault("game.verbosity", true)
	viper.SetDefault("game.time_between_rounds", 1*time.Second) // 0 to avoid any sleeping
}
