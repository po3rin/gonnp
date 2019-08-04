all: test e2e staticcheck

.PHONY: test
test:
	go test ./...

.PHONY: e2e
e2e:
	go test ./... -tags=e2e -v

.PHONY: staticcheck
staticcheck:
	sudo staticcheck ./...

.PHONY: cover
cover:
	go test -coverprofile=cover.out ./... && \
		go tool cover -html=cover.out
