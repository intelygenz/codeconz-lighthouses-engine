# Build the application
build:
	go build -o bin/lighthouses_aicontest

# Run the application
rungs:
	go run main.go -la=:3000

# Run bot 1
runbot1:
	go run ./examples/ranbot.go -bn=bot1 -la=:3001  -gs=localhost:3000

# Run bot 2
runbot2:
	go run ./examples/ranbot.go -bn=bot2 -la=:3002  -gs=localhost:3000

# Run tests
test:
	go test -v ./...

# Generate protobuf files
proto:
	protoc -I=./proto \
	--go_out=./coms \
	--go_opt=paths=source_relative \
	--go-grpc_out=./coms \
	--go-grpc_opt=paths=source_relative,require_unimplemented_servers=false \
	--py-out=./coms \
	--py_opt=paths=source_relative \
	--grpc_python_out=./coms \
	--grpc_python_opt=paths=source_relative \
	./proto/*.proto

.PHONY: build run test proto