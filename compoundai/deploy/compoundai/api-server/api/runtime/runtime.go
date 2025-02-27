package runtime

import (
	"fmt"

	"github.com/rs/zerolog/log"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/database"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/routes"
)

type runtime struct{}

var Runtime = runtime{}

func (r *runtime) StartServer(port int) {
	database.SetupDB()
	router := routes.SetupRouter()

	log.Info().Msgf("Starting CompoundAI API server on port %d", port)

	router.Run(fmt.Sprintf("0.0.0.0:%d", port))
}
