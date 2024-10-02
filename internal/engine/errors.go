package engine

type NoMorePlayersAcceptedError struct {
}

func (e *NoMorePlayersAcceptedError) Error() string {
	return "no more players accepted"
}
