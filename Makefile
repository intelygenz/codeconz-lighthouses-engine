SHELL := /bin/bash
GAME_BOTS    := 3
GAME_NETWORK := gamenw
SERVER_PORT  := 50051

# Build the application
build:
	go build -o bin/lighthouses_aicontest

# Run the application
rungs:
	go run ./cmd/main.go

# Run bot 1
runbotgo:
	go run ./examples/ranbot.go -bn=bot1 -la=:3001 -gs=:$(SERVER_PORT)

# Run py bot
runbotpy:
	python -m examples.randbot --bn python-bot1 --la=localhost:3001 --gs=localhost:$(SERVER_PORT)

# Run linter
lint:
	golangci-lint run

# Run tests
test:
	go test -v ./...

# full docker test spinning up $(GAME_BOTS)
docker-test: docker-net-up docker-build docker-game-simulation docker-destroy

docker-net-up:
	# creating docker network $(GAME_NETWORK)
	@docker network create $(GAME_NETWORK)

docker-net-down:
	# deleting docker network $(GAME_NETWORK)
	@docker network rm $(GAME_NETWORK)

docker-build:
	# building the game server & $(GAME_BOTS) bots
	@echo "==> building game server"
	@docker build -f ./docker/Dockerfile.game . -t game
	@for i in {1..$(GAME_BOTS)} ; do echo "==> building gobot$${i}" ; docker build -f ./docker/Dockerfile.gobot . --build-arg BOT_PORT=300$${i} --build-arg BOT_NAME=gobot$${i} -t gobot$${i} ; done
	@for i in {1..$(GAME_BOTS)} ; do echo "==> building pybot$${i}" ; docker build -f ./docker/Dockerfile.pybot . --build-arg BOT_PORT=300$${i} --build-arg BOT_NAME=pybot$${i} -t pybot$${i} ; done

docker-game-simulation-gobot:
	# simulating a game with $(GAME_BOTS) bots
	@docker run -d --rm --net $(GAME_NETWORK) --name game -v ./output:/app/output -p $(SERVER_PORT):$(SERVER_PORT) game
	@for i in {1..$(GAME_BOTS)} ; do docker run -d --rm --net $(GAME_NETWORK) --name gobot$${i} -p 300$${i}:300$${i} -e BOT_PORT=300$${i} -e BOT_NAME=gobot$${i} gobot$${i} ; done
	@docker logs -tf game
	@echo "==> game output files:"
	@ls -lanh ./output/
	# stopping docker containers
	@docker ps -a | awk '/(gobot|game)/ {print $$1}' | xargs --no-run-if-empty docker stop

docker-game-simulation-pybot:
	# simulating a game with $(GAME_BOTS) bots
	@docker run -d --rm --net $(GAME_NETWORK) --name game -v ./output:/app/output -p $(SERVER_PORT):$(SERVER_PORT) game
	@for i in {1..$(GAME_BOTS)} ; do docker run -d --rm --net $(GAME_NETWORK) --name pybot$${i} -p 300$${i}:300$${i} -e BOT_PORT=300$${i} -e BOT_NAME=pybot$${i} pybot$${i} ; done
	@docker logs -tf game
	@echo "==> game output files:"
	@ls -lanh ./output/
	# stopping docker containers
	@docker ps -a | awk '/(pybot|game)/ {print $$1}' | xargs --no-run-if-empty docker stop

docker-destroy:
	# stopping docker containers
	@docker ps -a | awk '/(pybot|gobot|game)/ {print $$1}' | xargs --no-run-if-empty docker stop
	# cleaning up all docker images
	@docker images --format '{{.Repository}}' | awk '/^(pybot|gobot|game)/ {print $$1}' | sort -r | xargs --no-run-if-empty docker rmi -f
	# deleting docker network $(GAME_NETWORK)
	@docker network rm -f $(GAME_NETWORK)

# Generate protobuf files
proto-go:
	protoc -I=./proto \
	--go_out=./internal/handler/coms \
	--go_opt=paths=source_relative \
	--go-grpc_out=./internal/handler/coms \
	--go-grpc_opt=paths=source_relative,require_unimplemented_servers=false \
	./proto/*.proto

proto-py:
	python3 -m grpc_tools.protoc -I=./proto \
	--python_out=./internal/handler/coms \
	--pyi_out=./internal/handler/coms \
	--grpc_python_out=./internal/handler/coms \
	./proto/*.proto

.PHONY: build rungs runbotgo runbotpy lint test docker-net-up docker-net-down docker-build docker-game-simulation-gobot docker-game-simulation-pybot docker-destroy proto-go proto-py
