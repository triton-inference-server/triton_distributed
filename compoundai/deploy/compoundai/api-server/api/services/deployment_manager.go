package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/common/utils"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/crds"
	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/models"
)

type deploymentManagementService struct{}

var DeploymentManagementService = deploymentManagementService{}

type DMSConfiguration struct {
	Version string      `json:"version"`
	Data    interface{} `json:"data"`
}

type DMSCreateRequest struct {
	Name          string                  `json:"name"`
	Namespace     string                  `json:"namespace"`
	ResourceType  crds.CustomResourceType `json:"type"`
	Configuration interface{}             `json:"configuration"`
}

type DMSResponseStatus struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type DMSCreateResponse struct {
	Id            string            `json:"id"`
	Status        DMSResponseStatus `json:"status"`
	Configuration interface{}       `json:"configuration"`
}

func (s *deploymentManagementService) Create(ctx context.Context, deploymentTarget *models.DeploymentTarget, deployOption *models.DeployOption) (*models.DeploymentTarget, error) {
	dmsHost, dmsPort, err := getDMSPortAndHost()
	if err != nil {
		log.Error().Msg(err.Error())
		return nil, err
	}

	url := fmt.Sprintf("http://%s:%s/v1/deployments", dmsHost, dmsPort)
	deployment, err := DeploymentService.Get(ctx, deploymentTarget.DeploymentId)

	if err != nil {
		log.Info().Msg("Could not find associated deployment")
		return nil, err
	}

	defer func() {
		if err != nil {
			s.Delete(ctx, deploymentTarget)
		}
	}()

	compoundAINimDeployment, compoundAINimRequest := s.transformToDMSRequestsV1alpha1(deployment, deploymentTarget)

	body, err := sendRequest(compoundAINimDeployment, url, http.MethodPost)
	if err != nil {
		return nil, err
	}
	var result DMSCreateResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println("Error unmarshaling:", err)
		return nil, err
	}
	deploymentTarget.KubeDeploymentId = result.Id

	body, err = sendRequest(compoundAINimRequest, url, http.MethodPost)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println("Error unmarshaling:", err)
		return nil, err
	}

	deploymentTarget.KubeRequestId = result.Id
	return deploymentTarget, nil
}

func (s *deploymentManagementService) Delete(ctx context.Context, deploymentTarget *models.DeploymentTarget) error {
	dmsHost, dmsPort, err := getDMSPortAndHost()
	if err != nil {
		log.Error().Msg(err.Error())
		return err
	}

	if deploymentTarget.KubeDeploymentId != "" {
		urlDeployment := fmt.Sprintf("http://%s:%s/v1/deployments/%s", dmsHost, dmsPort, deploymentTarget.KubeDeploymentId)
		_, err := sendRequest(nil, urlDeployment, http.MethodDelete)
		if err != nil {
			return err
		}
	}

	if deploymentTarget.KubeRequestId != "" {
		urlRequest := fmt.Sprintf("http://%s:%s/v1/deployments/%s", os.Getenv("DMS_HOST"), os.Getenv("DMS_PORT"), deploymentTarget.KubeRequestId)
		_, err := sendRequest(nil, urlRequest, http.MethodDelete)
		if err != nil {
			return err
		}
	}

	return nil
}

func (s *deploymentManagementService) transformToDMSRequestsV1alpha1(deployment *models.Deployment, deploymentTarget *models.DeploymentTarget) (compoundAINimDeployment DMSCreateRequest, compoundAINimRequest DMSCreateRequest) {
	translatedTag := s.translateCompoundAINimTagToRFC1123(deploymentTarget.CompoundAINimTag)

	compoundAINimDeployment = DMSCreateRequest{
		Name:         deployment.Name,
		Namespace:    deployment.KubeNamespace,
		ResourceType: crds.CompoundAINimDeployment,
		Configuration: crds.CompoundAINimDeploymentConfigurationV1Alpha1{
			Data: crds.CompoundAINimDeploymentData{
				CompoundAINim: translatedTag,
				Resources:     *deploymentTarget.Config.Resources,
			},
			Version: crds.ApiVersion,
		},
	}

	compoundAINimRequest = DMSCreateRequest{
		Name:         translatedTag,
		Namespace:    deployment.KubeNamespace,
		ResourceType: crds.CompoundAINimRequest,
		Configuration: crds.CompoundAINimRequestConfigurationV1Alpha1{
			Data: crds.CompoundAINimRequestData{
				BentoTag: deploymentTarget.CompoundAINimTag,
			},
			Version: crds.ApiVersion,
		},
	}
	return
}

func getDMSPortAndHost() (string, string, error) {
	dmsHost, err := utils.MustGetEnv("DMS_HOST")
	if err != nil {
		return "", "", err
	}

	dmsPort, err := utils.MustGetEnv("DMS_PORT")
	if err != nil {
		return "", "", err
	}

	return dmsHost, dmsPort, nil
}

/**
 * Translates a CompoundAI Nim tag to a valid RFC 1123 DNS label.
 *
 * This function makes the following modifications to the input string:
 * 1. Replaces all ":" characters with "--" because colons are not permitted in DNS labels.
 * 2. If the resulting string exceeds the 63-character limit imposed by RFC 1123, it truncates
 *    the string to 63 characters.
 *
 * @param {string} tag - The original CompoundAI Nim tag that needs to be converted.
 * @returns {string} - A string that complies with the RFC 1123 DNS label format.
 *
 * Example:
 *   Input: "nim:latest"
 *   Output: "nim--latest"
 */
func (s *deploymentManagementService) translateCompoundAINimTagToRFC1123(tag string) string {
	translated := strings.ReplaceAll(tag, ":", "--")

	// If the length exceeds 63 characters, truncate it
	if len(translated) > 63 {
		translated = translated[:63]
	}

	return translated
}

func sendRequest(payload interface{}, url string, method string) ([]byte, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	req, err := http.NewRequest(method, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("received non-OK response: %v, %s", resp.Status, body)
	}

	return body, nil
}
