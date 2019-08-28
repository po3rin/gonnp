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

.PHONY: mnist
mnist:
	go test -v --tags=e2e ./... -run TestMNIST

.PHONY: simplecbow
simplecbow:
	go test -v --tags=e2e ./... -run TestSimpleCBOW

.PHONY: cbow
cbow:
	go test -v --tags=e2e ./... -run TestCBOW

.PHONY: profile
profile:
	# should change loop num. maxIters=10, maxEpoch=10.
	go build dev/profile/cbow/main.go && \
		./main && \
		go tool pprof main cpu.pprof
