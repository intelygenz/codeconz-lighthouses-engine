# Build the Go binary
FROM golang:1.22-alpine AS builder
ARG GAME_PORT=50051
WORKDIR /app
COPY . ./
RUN go mod download && \
  CGO_ENABLED=0 GOOS=linux go build -o game cmd/main.go

FROM alpine:3.20.3
WORKDIR /app
COPY . ./
COPY --from=builder /app/game ./
EXPOSE ${GAME_PORT}
RUN adduser -h /app -H -s /sbin/nologin -D -u 10000 gamesrv && \
  chown -R gamesrv:gamesrv /app
USER gamesrv
# Command to run the Go binary
CMD ["./game"]
