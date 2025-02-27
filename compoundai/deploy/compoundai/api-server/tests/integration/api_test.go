package integration

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/database"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/runtime"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/services"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/tests/integration/fixtures"
)

const (
	port                = 9999
	expectedStatusOkMsg = "Expected status code 200"
)

var apiServerUrl = fmt.Sprintf("http://localhost:%d", port)
var client = CompoundAIClient{
	url: apiServerUrl,
}

type ApiServerSuite struct {
	suite.Suite
}

func TestApiServerSuite(t *testing.T) {
	suite.Run(t, new(ApiServerSuite))
}

// run once, before test suite methods
func (s *ApiServerSuite) SetupSuite() {
	log.Info().Msgf("Starting suite...")
	_, err := testContainers.CreatePostgresContainer()
	if err != nil {
		s.T().FailNow()
	}

	// Setup server
	go func() {
		// Mute all logs for this goroutine
		gin.DefaultWriter = io.Discard
		runtime.Runtime.StartServer(port)
	}()

	s.waitUntilReady()
}

func (s *ApiServerSuite) waitUntilReady() {
	url := fmt.Sprintf("%s/healthz", apiServerUrl)
	for {
		resp, err := http.Get(url)
		if err == nil && resp.StatusCode == http.StatusOK {
			log.Info().Msg("CompoundAI API server is running")
			return // Server is ready
		}
		log.Info().Msgf("Waiting 500ms before checking /healthz again")
		time.Sleep(500 * time.Millisecond) // Wait before retrying
	}
}

// run once, after test suite methods
func (s *ApiServerSuite) TearDownSuite() {
	testContainers.TearDownPostgresContainer()
}

// run after each test
func (s *ApiServerSuite) TearDownTest() {
	ctx := context.Background()
	db := database.DatabaseUtil.GetDBSession(ctx)

	if err := db.Unscoped().Where("true").Delete(&models.Deployment{}).Error; err != nil {
		s.T().Fatalf("Failed to delete records from deployment table: %v", err)
	}

	if err := db.Unscoped().Where("true").Delete(&models.Cluster{}).Error; err != nil {
		s.T().Fatalf("Failed to delete records from cluster table: %v", err)
	}
}

func (s *ApiServerSuite) TestCreateCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, clusterFullSchema := client.CreateCluster(s.T(), cluster)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content (optional)
	assert.Equal(s.T(), clusterFullSchema.Description, cluster.Description)
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), cluster.KubeConfig)
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestGetCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Get Cluster
	resp, clusterFullSchema := client.GetCluster(s.T(), cluster.Name)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content
	assert.Equal(s.T(), clusterFullSchema.Description, cluster.Description)
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), cluster.KubeConfig)
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestGetUnknownClusterFails() {
	resp, _ := client.GetCluster(s.T(), "unknown")
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestGetMultipleClusters() {
	cluster1 := fixtures.DefaultCreateClusterSchema()
	cluster1.Name = "c1"
	resp, _ := client.CreateCluster(s.T(), cluster1)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	cluster2 := fixtures.DefaultCreateClusterSchema()
	cluster2.Name = "c2"
	resp, _ = client.CreateCluster(s.T(), cluster2)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	req := fixtures.DefaultListQuerySchema()
	resp, clusterListSchema := client.GetClusterList(s.T(), req)

	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	for _, item := range clusterListSchema.Items {
		assert.Contains(s.T(), []string{"c1", "c2"}, item.Name, expectedStatusOkMsg)
	}
}

func (s *ApiServerSuite) TestUpdateCluster() {
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	req := fixtures.DefaultUpdateClusterSchema()
	d := "Nemo"
	kc := "KcNemo"
	req.Description = &d
	req.KubeConfig = &kc

	resp, clusterFullSchema := client.UpdateCluster(s.T(), cluster.Name, req)

	// Verify the response status code
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Additional checks on response content (optional)
	assert.Equal(s.T(), clusterFullSchema.Description, *(req.Description))
	assert.Equal(s.T(), *(clusterFullSchema.KubeConfig), *(req.KubeConfig))
	assert.Equal(s.T(), clusterFullSchema.Name, cluster.Name)
}

func (s *ApiServerSuite) TestCreateDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, deploymentSchema := client.GetDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	assert.Equal(s.T(), deployment.Name, deploymentSchema["name"])
	assert.Equal(s.T(), *(deployment.Description), deploymentSchema["description"])
	assert.Equal(s.T(), string(schemas.DeploymentStatusNonDeployed), deploymentSchema["status"])
}

func (s *ApiServerSuite) TestCreateDeploymentUnknownClusterFails() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ := client.CreateDeployment(s.T(), "unknown", deployment)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestUpdateDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Create deployment
	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Update deployment
	updateTarget := fixtures.DefaultCreateDeploymentTargetSchema()
	updateTarget.Bento = "2025"
	updateTarget.BentoRepository = "compoundai"

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	updatedDescription := "new description"
	updateDeployment.Description = &updatedDescription

	updateDeployment.Targets = []*schemas.CreateDeploymentTargetSchema{
		updateTarget,
	}

	resp, deploymentSchema := client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Validate fields
	assert.Equal(s.T(), deployment.Name, deploymentSchema["name"])
	assert.Equal(s.T(), updatedDescription, deploymentSchema["description"])

	// Todo: once Deployment Schema is available make this test more rigorous
	ctx := context.Background()
	revisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 2, len(revisions))

	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(activeRevisions))
}

func (s *ApiServerSuite) TestUpdateDeploymentWithoutDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Create deployment
	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Update deployment
	updateTarget := fixtures.DefaultCreateDeploymentTargetSchema()
	updateTarget.Config.KubeResourceUid = "abc123"
	updateTarget.Config.KubeResourceVersion = "alphav1"

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	updateDeployment.Targets = []*schemas.CreateDeploymentTargetSchema{
		updateTarget,
	}
	updateDeployment.DoNotDeploy = true

	resp, deploymentSchema := client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Validate fields
	assert.Equal(s.T(), deployment.Name, deploymentSchema["name"])

	// Todo: once Deployment Schema is available make this test more rigorous

	// Updating without deployment does not deactivate any deployment revision
	ctx := context.Background()
	revisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(revisions))

	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(activeRevisions))

	deploymentTargets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{})
	if err != nil {
		s.T().Fatalf("Could not fetch targets: %s", err.Error())
	}

	assert.Equalf(s.T(), 1, len(deploymentTargets), "More deployment targets than expected")

	assert.Equal(s.T(), updateTarget.Config.KubeResourceUid, deploymentTargets[0].Config.KubeResourceUid)
	assert.Equal(s.T(), updateTarget.Config.KubeResourceVersion, deploymentTargets[0].Config.KubeResourceVersion)
}

func (s *ApiServerSuite) TestUpdateDeploymentUnknownClusterFails() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ := client.UpdateDeployment(s.T(), "unknown", "default", "unknown", updateDeployment)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestUpdateDeploymentUnknownDeploymentFails() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	// Create cluster
	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, "default", "unknown", updateDeployment)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestTerminateDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	status_ := schemas.DeploymentRevisionStatusActive
	activeRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(activeRevisions))

	// Terminate deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	status_ = schemas.DeploymentRevisionStatusInactive
	inactiveRevisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{
		Status: &status_,
	})
	if err != nil {
		s.T().Fatalf("Could not fetch revisions: %s", err.Error())
	}

	assert.Equal(s.T(), 1, len(inactiveRevisions))
}

func (s *ApiServerSuite) TestTerminateNonExistingDeployment() {
	resp, _ := client.TerminateDeployment(s.T(), "nonexistent-cluster", "nonexistent-namespace", "nonexistent-deployment")
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestTerminateNonIncorrectDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	resp, _ = client.TerminateDeployment(s.T(), "nonexistent-cluster", deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")

	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, "nonexistent-namespace", deployment.Name)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")

	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, "nonexistent-deployment")
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestDeleteDeactivatedDeployment() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Terminate the deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Delete the deactivated deployment
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Check that there are no remaining deployment entities
	ctx := context.Background()
	d, r, t, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Could not fetch deployment entities snapshot: %s", err.Error())
	}
	assert.Equal(s.T(), 0, len(d))
	assert.Equal(s.T(), 0, len(r))
	assert.Equal(s.T(), 0, len(t))
}

func (s *ApiServerSuite) TestDeleteActiveDeploymentFails() {
	server := fixtures.CreateMockDMSServer(s.T())
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	// Attempt to delete the active deployment
	resp, _ = client.DeleteDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")
}

func (s *ApiServerSuite) TestUpdateDeploymentWithDMSErrorDoesNotChangeDB() {
	server := fixtures.CreateMockDMSServer(s.T()) // Does not throw error initially
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}
	server.Throws(true)

	// Attempt to update the deployment
	updateDeployment := fixtures.DefaultUpdateDeploymentSchema()
	resp, _ = client.UpdateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name, updateDeployment)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")

	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func (s *ApiServerSuite) TestTerminateDeploymentWithDMSErrorDoesNotChangeDB() {
	server := fixtures.CreateMockDMSServer(s.T()) // Does not throw error initially
	defer server.Close()

	cluster := fixtures.DefaultCreateClusterSchema()
	resp, _ := client.CreateCluster(s.T(), cluster)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	deployment := fixtures.DefaultCreateDeploymentSchema()
	resp, _ = client.CreateDeployment(s.T(), cluster.Name, deployment)
	assert.Equal(s.T(), http.StatusOK, resp.StatusCode, expectedStatusOkMsg)

	ctx := context.Background()
	d1, r1, t1, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}
	server.Throws(true)

	// Attempt to terminate the deployment
	resp, _ = client.TerminateDeployment(s.T(), cluster.Name, deployment.KubeNamespace, deployment.Name)
	assert.Equal(s.T(), http.StatusBadRequest, resp.StatusCode, "Expected status code 400")

	// Verify DB state remains unchanged
	d2, r2, t2, err := getDeploymentEntitiesSnapshot(ctx)
	if err != nil {
		s.T().Fatalf("Failed to get snapshot %s", err.Error())
	}

	assert.True(s.T(), compareDeployments(d1, d2))
	assert.True(s.T(), compareDeploymentRevisions(r1, r2))
	assert.True(s.T(), compareDeploymentTargets(t1, t2))
}

func compareDeployments(slice1, slice2 []*models.Deployment) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected deployment: %+v", slice1[i])
			log.Info().Msgf("Actual deployment: %+v", slice2[i])
			return false
		}
	}

	return true
}

func compareDeploymentRevisions(slice1, slice2 []*models.DeploymentRevision) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected revision: %+v", slice1[i])
			log.Info().Msgf("Actual revision: %+v", slice2[i])
			return false
		}
	}

	return true
}

func compareDeploymentTargets(slice1, slice2 []*models.DeploymentTarget) bool {
	// Check if lengths are equal
	if len(slice1) != len(slice2) {
		return false
	}

	// Compare each element using reflect.DeepEqual
	for i := range slice1 {
		if !reflect.DeepEqual(slice1[i], slice2[i]) {
			log.Info().Msgf("Expected target: %+v", slice1[i])
			log.Info().Msgf("Actual target: %+v", slice2[i])
			return false
		}
	}

	return true
}

func getDeploymentEntitiesSnapshot(ctx context.Context) ([]*models.Deployment, []*models.DeploymentRevision, []*models.DeploymentTarget, error) {
	deployments, _, err := services.DeploymentService.List(ctx, services.ListDeploymentOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	revisions, _, err := services.DeploymentRevisionService.List(ctx, services.ListDeploymentRevisionOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	targets, _, err := services.DeploymentTargetService.List(ctx, services.ListDeploymentTargetOption{})
	if err != nil {
		return nil, nil, nil, err
	}

	return deployments, revisions, targets, nil
}
