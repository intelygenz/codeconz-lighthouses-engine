package bootstrap

import (
	"fmt"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/game"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
	"net"
	"time"
)

type Bootstrap struct {
}

func NewBootstrap() *Bootstrap {
	return &Bootstrap{}
}

func (b *Bootstrap) Run() {
	b.initializeConfiguration()

	fmt.Println("Game server starting on", viper.GetString("game.listen_address"))

	if viper.GetString("game.listen_address") == "" {
		panic("addr is required")
	}

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

	fmt.Println("players joining is closed, now the game starts!!!")

	ge.StartGame()

	fmt.Println("Game finished!!!")
}

func (b *Bootstrap) initializeConfiguration() {
	viper.SetDefault("game.listen_address", ":50051")
	viper.SetDefault("game.join_timeout", 10*time.Second)
	viper.SetDefault("game.turn_request_timeout", 100*time.Millisecond)
	viper.SetDefault("game.turns", 20)
	viper.SetDefault("game.board_path", "/Users/dovixman/Workspace/Work/Programs/SecretEvent/lighthouses_aicontest/maps/square_xl.txt")
}
