test:
	go test ./...

staticcheck:
	sudo staticcheck ./...

cover:
	go test -coverprofile=cover.out ./... && \
		go tool cover -html=cover.out
