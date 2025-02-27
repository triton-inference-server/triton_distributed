package integration

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"testing"

	"github.com/triton-inference-server/triton_distributed/deploy/compoundai/api-server/api/schemas"
)

/**
	This file exposes a series of helper functions to utilize the CompoundAI API server
**/

type CompoundAIClient struct {
	url string
}

func (c *CompoundAIClient) CreateCluster(t *testing.T, s schemas.CreateClusterSchema) (*http.Response, schemas.ClusterFullSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	resp, err := http.Post(c.url+"/api/v1/clusters", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to create cluster %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, clusterFullSchema
}

func (c *CompoundAIClient) UpdateCluster(t *testing.T, name string, s schemas.UpdateClusterSchema) (*http.Response, schemas.ClusterFullSchema) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create the PATCH request with JSON data
	req, err := http.NewRequest(http.MethodPatch, c.url+"/api/v1/clusters/"+name, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed create update request %s", err.Error())
	}

	// Set the appropriate headers
	req.Header.Set("Content-Type", "application/json")

	// Create an HTTP client and send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to update cluster %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, clusterFullSchema
}

func (c *CompoundAIClient) GetCluster(t *testing.T, name string) (*http.Response, *schemas.ClusterFullSchema) {
	resp, err := http.Get(c.url + "/api/v1/clusters/" + name)
	if err != nil {
		t.Fatalf("Failed to get cluster %s", err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var clusterFullSchema schemas.ClusterFullSchema
	if err = json.Unmarshal(respBody, &clusterFullSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &clusterFullSchema
}

func encodeListQuerySchema(query schemas.ListQuerySchema) string {
	params := url.Values{}
	params.Set("start", strconv.FormatUint(uint64(query.Start), 10))
	params.Set("count", strconv.FormatUint(uint64(query.Count), 10))
	if query.Search != nil {
		params.Set("search", *query.Search)
	}
	params.Set("q", query.Q)
	return params.Encode()
}

func (c *CompoundAIClient) GetClusterList(t *testing.T, s schemas.ListQuerySchema) (*http.Response, *schemas.ClusterListSchema) {
	form := encodeListQuerySchema(s)

	resp, err := http.Get(c.url + "/api/v1/clusters?" + form)
	if err != nil {
		t.Fatalf("Failed to create cluster %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var clusterListSchema schemas.ClusterListSchema
	if err = json.Unmarshal(respBody, &clusterListSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &clusterListSchema
}

func (c *CompoundAIClient) CreateDeployment(t *testing.T, clusterName string, s schemas.CreateDeploymentSchema) (*http.Response, any) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	resp, err := http.Post(c.url+"/api/v1/clusters/"+clusterName+"/deployments", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to create deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema any
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, deploymentSchema
}

func (c *CompoundAIClient) UpdateDeployment(t *testing.T, clusterName, namespace string, deploymentName string, s schemas.UpdateDeploymentSchema) (*http.Response, map[string]any) {
	body, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Create the PATCH request with JSON data
	req, err := http.NewRequest(http.MethodPatch, c.url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName, bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed create update request %s", err.Error())
	}

	// Set the appropriate headers
	req.Header.Set("Content-Type", "application/json")

	// Create an HTTP client and send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to update deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deploymentSchema map[string]any
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, deploymentSchema
}

func (c *CompoundAIClient) GetDeployment(t *testing.T, clusterName, namespace string, deploymentName string) (*http.Response, map[string]any) {
	resp, err := http.Get(c.url + "/api/v1/clusters/" + clusterName + "/namespaces/" + namespace + "/deployments/" + deploymentName)
	if err != nil {
		t.Fatalf("Failed to get deployment %s", err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentSchema map[string]any
	if err = json.Unmarshal(respBody, &deploymentSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, deploymentSchema
}

func (c *CompoundAIClient) GetDeploymentList(t *testing.T, clusterName string, query schemas.ListQuerySchema) (*http.Response, *any) {
	form := encodeListQuerySchema(query)

	resp, err := http.Get(c.url + "/api/v1/clusters/" + clusterName + "/deployments?" + form)
	if err != nil {
		t.Fatalf("Failed to get deployments %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var deploymentListSchema any
	if err = json.Unmarshal(respBody, &deploymentListSchema); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, &deploymentListSchema
}

func (c *CompoundAIClient) SyncDeploymentStatus(t *testing.T, clusterName, deploymentName string) (*http.Response, any) {
	resp, err := http.Post(c.url+"/api/v1/clusters/"+clusterName+"/deployments/"+deploymentName+"/sync_status", "application/json", nil)
	if err != nil {
		t.Fatalf("Failed to sync deployment status %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	var statusResponse any
	if err = json.Unmarshal(respBody, &statusResponse); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, statusResponse
}

func (c *CompoundAIClient) TerminateDeployment(t *testing.T, clusterName, namespace string, deploymentName string) (*http.Response, any) {
	resp, err := http.Post(c.url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName+"/terminate", "application/json", nil)
	if err != nil {
		t.Fatalf("Failed to terminate deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var terminateResponse any
	if err = json.Unmarshal(respBody, &terminateResponse); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, terminateResponse
}

func (c *CompoundAIClient) DeleteDeployment(t *testing.T, clusterName, namespace string, deploymentName string) (*http.Response, any) {
	req, err := http.NewRequest(http.MethodDelete, c.url+"/api/v1/clusters/"+clusterName+"/namespaces/"+namespace+"/deployments/"+deploymentName, nil)
	if err != nil {
		t.Fatalf("Failed to create delete request %s", err.Error())
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Failed to delete deployment %s", err.Error())
	}
	defer resp.Body.Close()

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var deleteResponse any
	if err = json.Unmarshal(respBody, &deleteResponse); err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	return resp, deleteResponse
}
