package main

import "github.com/jonasdacruz/lighthouses_aicontest/cmd/bootstrap"

func main() {
	boot := bootstrap.NewBootstrap()

	boot.Run()

	/*
		island := board.NewBoard("/Users/dovixman/Workspace/Work/Programs/SecretEvent/lighthouses_aicontest/maps/island.txt")
		island.PrettyPrintMap()
		fmt.Println()
		island.PrettyPrintBoolMap()

		for i := 0; i < 10; i++ {
			island.CalcEnergy()
			island.PrettyPrintMap()
			fmt.Println()
			fmt.Println()
			fmt.Println()
		}
	*/

	/*
		game := game.NewGame("/Users/dovixman/Workspace/Work/Programs/SecretEvent/lighthouses_aicontest/maps/island.txt", 100)
		game.StartGame()
	*/
}
