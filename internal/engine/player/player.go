package player

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/twpayne/go-geom"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	ActionPass = iota
	ActionMove
	ActionAttack
	ActionConnect
)

type Action struct {
	Action      int
	Destination geom.Coord
	Energy      int
}

type Turn struct {
	Position    geom.Coord
	Score       int
	Energy      int
	View        [][]int
	Lighthouses []*lighthouse.Lighthouse
}

type Player struct {
	ServerAddress     string                   `json:"-"`
	ID                int                      `json:"id"`
	Name              string                   `json:"name"`
	Score             int                      `json:"score"`
	Energy            int                      `json:"energy"`
	Position          geom.Coord               `json:"position"`
	LighthouseKeys    []*lighthouse.Lighthouse `json:"-"`
	LighthouseKeyIds  []int                    `json:"keys"`
	gameServiceClient coms.GameServiceClient   `json:"-"`
}

func NewPlayer(serverAddress string, name string) *Player {
	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(serverAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	return &Player{
		ServerAddress:     serverAddress,
		ID:                -1,
		Name:              name,
		Score:             0,
		Energy:            0,
		Position:          geom.Coord{0, 0},
		LighthouseKeys:    make([]*lighthouse.Lighthouse, 0),
		gameServiceClient: coms.NewGameServiceClient(grpcClient),
	}
}

func (p *Player) SetPosition(x, y int) {
	p.Position = geom.Coord{float64(x), float64(y)}
}

func (p *Player) GetPosition() (x, y int) {
	return int(p.Position.X()), int(p.Position.Y())
}

func (p *Player) AddLighthouseKey(lighthouse *lighthouse.Lighthouse) {
	if !p.LighthouseKeyExists(lighthouse) {
		p.LighthouseKeys = append(p.LighthouseKeys, lighthouse)
	}
}

func (p *Player) LighthouseKeyExists(lighthouse *lighthouse.Lighthouse) bool {
	for _, l := range p.LighthouseKeys {
		if l.Position.Equal(geom.XY, lighthouse.Position) {
			return true
		}
	}
	return false
}

func (p *Player) RemoveLighthouseKey(lighthouse lighthouse.Lighthouse) {
	for i, l := range p.LighthouseKeys {
		if l.Position.Equal(geom.XY, lighthouse.Position) {
			p.LighthouseKeys = append(p.LighthouseKeys[:i], p.LighthouseKeys[i+1:]...)
			break
		}
	}
}

func (p *Player) GenerateLighthouseKeysIds() {
	p.LighthouseKeyIds = make([]int, 0)
	for _, l := range p.LighthouseKeys {
		p.LighthouseKeyIds = append(p.LighthouseKeyIds, l.ID)
	}
}
