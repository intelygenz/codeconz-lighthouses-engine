package engine

type NoMorePlayersAcceptedError struct {
}

func (e *NoMorePlayersAcceptedError) Error() string {
	return "No more players accepted"
}
