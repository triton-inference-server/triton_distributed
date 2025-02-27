package main

import "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/runtime"

const (
	port = 8181
)

func main() {
	runtime.Runtime.StartServer(port)
}
