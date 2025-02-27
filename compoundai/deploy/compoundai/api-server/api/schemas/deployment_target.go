package schemas

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

type DeploymentTargetType string

const (
	DeploymentTargetTypeStable DeploymentTargetType = "stable"
	DeploymentTargetTypeCanary DeploymentTargetType = "canary"
)

var DeploymentTargetTypeAddrs = map[DeploymentTargetType]string{
	DeploymentTargetTypeStable: "stb",
	DeploymentTargetTypeCanary: "cnr",
}

type DeploymentTargetHPAConf struct {
	CPU         *int32  `json:"cpu,omitempty"`
	GPU         *int32  `json:"gpu,omitempty"`
	Memory      *string `json:"memory,omitempty"`
	QPS         *int64  `json:"qps,omitempty"`
	MinReplicas *int32  `json:"min_replicas,omitempty"`
	MaxReplicas *int32  `json:"max_replicas,omitempty"`
}

type DeploymentStrategy string

const (
	DeploymentStrategyRollingUpdate               DeploymentStrategy = "RollingUpdate"
	DeploymentStrategyRecreate                    DeploymentStrategy = "Recreate"
	DeploymentStrategyRampedSlowRollout           DeploymentStrategy = "RampedSlowRollout"
	DeploymentStrategyBestEffortControlledRollout DeploymentStrategy = "BestEffortControlledRollout"
)

type DeploymentTargetConfig struct {
	KubeResourceUid                        string                   `json:"kubeResourceUid"`
	KubeResourceVersion                    string                   `json:"kubeResourceVersion"`
	Resources                              *Resources               `json:"resources"`
	HPAConf                                *DeploymentTargetHPAConf `json:"hpa_conf,omitempty"`
	EnableIngress                          *bool                    `json:"enable_ingress,omitempty"`
	EnableStealingTrafficDebugMode         *bool                    `json:"enable_stealing_traffic_debug_mode,omitempty"`
	EnableDebugMode                        *bool                    `json:"enable_debug_mode,omitempty"`
	EnableDebugPodReceiveProductionTraffic *bool                    `json:"enable_debug_pod_receive_production_traffic,omitempty"`
	DeploymentStrategy                     *DeploymentStrategy      `json:"deployment_strategy,omitempty"`
}

type CreateDeploymentTargetSchema struct {
	BentoRepository string                  `json:"bento_repository"`
	Bento           string                  `json:"bento"`
	Config          *DeploymentTargetConfig `json:"config"`
}

func (c *DeploymentTargetConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}

	var data []byte
	switch v := value.(type) {
	case string:
		data = []byte(v)
	case []byte:
		data = v
	default:
		return fmt.Errorf("unsupported type: %T", value)
	}

	return json.Unmarshal(data, c)
}

func (c *DeploymentTargetConfig) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
