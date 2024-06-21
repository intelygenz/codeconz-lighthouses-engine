package bootstrap

import (
	"flag"
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

	listenAddress := flag.String("la", "", "game server listen address")
	flag.Parse()

	fmt.Println("Game server starting on", *listenAddress)

	if *listenAddress == "" {
		panic("addr is required")
	}

	lis, err := net.Listen("tcp", *listenAddress)
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(handler.UnaryLoggingInterceptor),
		grpc.StreamInterceptor(handler.StreamLoggingInterceptor),
	)

	ge := game.NewGame("/Users/dovixman/Workspace/Work/Programs/SecretEvent/lighthouses_aicontest/maps/square_xl.txt", 10)
	gs := handler.NewGameServer(ge)

	coms.RegisterGameServiceServer(grpcServer, gs)

	go func() {
		<-time.After(viper.GetDuration("game.jointimeout"))
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
	viper.SetDefault("game.jointimeout", 10*time.Second)
}
