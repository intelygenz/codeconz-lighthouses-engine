# Build the application
build:
	go build -o bin/lighthouses_aicontest

# Run the application
rungs:
	go run ./cmd/main.go

# Run bot 1
runbot1:
	go run ./examples/ranbot.go -bn=bot1 -la=:3001 -gs=:50051

# Run bot 2
runbot2:
	go run ./examples/ranbot.go -bn=bot2 -la=:3002 -gs=:50051

# Run bot 3
runbot3:
	go run ./examples/ranbot.go -bn=bot3 -la=:3003 -gs=:50051

# Run linter
lint:
	golangci-lint run

# Run tests
test:
	go test -v ./...

# Generate protobuf files
proto:
	protoc -I=./proto \
	--go_out=./internal/handler/coms \
	--go_opt=paths=source_relative \
	--go-grpc_out=./internal/handler/coms \
	--go-grpc_opt=paths=source_relative,require_unimplemented_servers=false \
	--python_out=./internal/handler/coms \
	--pyi_out=./internal/handler/coms \
	--grpclib_python_out=./internal/handler/coms \
	./proto/*.proto

.PHONY: build run test proto
