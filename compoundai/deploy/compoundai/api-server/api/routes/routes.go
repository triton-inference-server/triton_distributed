package routes

import (
	"github.com/gin-gonic/gin"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/controllers"
)

func SetupRouter() *gin.Engine {
	router := gin.Default()

	baseGroup := router.Group("")
	createK8sRoutes(baseGroup)

	apiGroup := router.Group("/api/v1")
	apiGroupV2 := router.Group("/api/v2")

	/* Start V1 APIs */
	createClusterRoutes(apiGroup)

	// These routes are proxied to NDS
	createCompoundAINimRoutes(apiGroup)

	createMiscellaneousRoutes(apiGroup)
	createMockedRoutes(apiGroup)
	/* End V1 APIs */

	/* Start V2 APIs */
	deploymentRoutesV2(apiGroupV2)
	/* End V2 APIs */

	return router
}

func createK8sRoutes(grp *gin.RouterGroup) {
	healthGroup := grp.Group("/healthz")
	healthGroup.GET("", controllers.HealthController.Get)

	readyGroup := grp.Group("/readyz")
	readyGroup.GET("", controllers.HealthController.Get)
}

func createClusterRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/clusters")

	resourceGrp := grp.Group("/:clusterName")

	resourceGrp.GET("", controllers.ClusterController.Get)

	resourceGrp.PATCH("", controllers.ClusterController.Update)

	grp.GET("", controllers.ClusterController.List)

	grp.POST("", controllers.ClusterController.Create)

	deploymentRoutes(resourceGrp)
}

func deploymentRoutes(grp *gin.RouterGroup) {
	namespacedGrp := grp.Group("/namespaces/:kubeNamespace/deployments")
	grp = grp.Group("/deployments")

	resourceGrp := namespacedGrp.Group("/:deploymentName")

	resourceGrp.GET("", controllers.DeploymentController.Get)

	resourceGrp.PATCH("", controllers.DeploymentController.Update)

	resourceGrp.POST("/sync_status", controllers.DeploymentController.SyncStatus)

	resourceGrp.POST("/terminate", controllers.DeploymentController.Terminate)

	resourceGrp.DELETE("", controllers.DeploymentController.Delete)

	// resourceGrp.GET("/terminal_records", controllers.DeploymentController.ListTerminalRecords)

	grp.GET("", controllers.DeploymentController.ListClusterDeployments)

	grp.POST("", controllers.DeploymentController.Create)

	// deploymentRevisionRoutes(resourceGrp)
}

func deploymentRoutesV2(grp *gin.RouterGroup) {
	grp = grp.Group("/deployments")
	grp.POST("", controllers.DeploymentController.CreateV2)

	resourceGrp := grp.Group("/:deploymentName")
	resourceGrp.GET("", controllers.DeploymentController.GetV2)
}

func createCompoundAINimRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/bento_repositories")

	resourceGrp := grp.Group("/:bentoRepositoryName")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.ProxyController.ReverseProxy)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)

	compoundAINimVersionRoutes(resourceGrp)
}

func compoundAINimVersionRoutes(grp *gin.RouterGroup) {
	grp = grp.Group("/bentos")

	resourceGrp := grp.Group("/:version")

	resourceGrp.GET("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status_syncing_at", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/update_image_build_status", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/models", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/deployments", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/start_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_multipart_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/complete_multipart_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_upload_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/presign_download_url", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/start_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PATCH("/finish_upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.PUT("/upload", controllers.ProxyController.ReverseProxy)

	resourceGrp.GET("/download", controllers.ProxyController.ReverseProxy)

	grp.GET("", controllers.ProxyController.ReverseProxy)

	grp.POST("", controllers.ProxyController.ReverseProxy)
}

func createMiscellaneousRoutes(grp *gin.RouterGroup) {
	versionGrp := grp.Group("/version")

	versionGrp.GET("", controllers.VersionController.Get)
}

// Legacy APIs used by the CLI
func createMockedRoutes(grp *gin.RouterGroup) {
	grp.GET("auth/current", controllers.MockedController.GetDefaultUser)
	grp.GET("current_org", controllers.MockedController.GetDefaultOrg)
}
