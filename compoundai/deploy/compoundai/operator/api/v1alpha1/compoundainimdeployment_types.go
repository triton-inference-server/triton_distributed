/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	compounaiCommon "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	CompoundAIDeploymentConditionTypeAvailable                 = "Available"
	CompoundAIDeploymentConditionTypeCompoundAINimFound        = "CompoundAINimFound"
	CompoundAIDeploymentConditionTypeCompoundAINimRequestFound = "CompoundAINimRequestFound"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CompoundAINimDeploymentSpec defines the desired state of CompoundAINimDeployment
type CompoundAINimDeploymentSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Annotations map[string]string `json:"annotations,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`

	CompoundAINim string `json:"compoundAINim"`

	Resources        *compounaiCommon.Resources `json:"resources,omitempty"`
	Autoscaling      *Autoscaling               `json:"autoscaling,omitempty"`
	Envs             []corev1.EnvVar            `json:"envs,omitempty"`
	EnvFromSecret    *string                    `json:"envFromSecret,omitempty"`
	PVC              *PVC                       `json:"pvc,omitempty"`
	RunMode          *RunMode                   `json:"runMode,omitempty"`
	ExternalServices map[string]ExternalService `json:"externalServices,omitempty"`

	Ingress IngressSpec `json:"ingress,omitempty"`

	MonitorExporter *compounaiCommon.MonitorExporterSpec `json:"monitorExporter,omitempty"`

	// +optional
	ExtraPodMetadata *compounaiCommon.ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	// +optional
	ExtraPodSpec *compounaiCommon.ExtraPodSpec `json:"extraPodSpec,omitempty"`

	LivenessProbe  *corev1.Probe `json:"livenessProbe,omitempty"`
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
}

type RunMode struct {
	Standalone *bool `json:"standalone,omitempty"`
}

type ExternalService struct {
	DeploymentSelectorKey   string `json:"deploymentSelectorKey,omitempty"`
	DeploymentSelectorValue string `json:"deploymentSelectorValue,omitempty"`
}

type IngressTLSSpec struct {
	SecretName string `json:"secretName,omitempty"`
}

type IngressSpec struct {
	Enabled           bool              `json:"enabled,omitempty"`
	UseVirtualService *bool             `json:"useVirtualService,omitempty"`
	HostPrefix        *string           `json:"hostPrefix,omitempty"`
	Annotations       map[string]string `json:"annotations,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
	TLS               *IngressTLSSpec   `json:"tls,omitempty"`
}

// CompoundAINimDeploymentStatus defines the observed state of CompoundAINimDeployment
type CompoundAINimDeploymentStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	Conditions []metav1.Condition `json:"conditions"`

	PodSelector map[string]string `json:"podSelector,omitempty"`
}

//+genclient
//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:storageversion
//+kubebuilder:printcolumn:name="Bento",type="string",JSONPath=".spec.bento",description="Bento"
//+kubebuilder:printcolumn:name="Available",type="string",JSONPath=".status.conditions[?(@.type=='Available')].status",description="Available"
//+kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// CompoundAINimDeployment is the Schema for the compoundainimdeployments API
type CompoundAINimDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CompoundAINimDeploymentSpec   `json:"spec,omitempty"`
	Status CompoundAINimDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// CompoundAINimDeploymentList contains a list of CompoundAINimDeployment
type CompoundAINimDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CompoundAINimDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CompoundAINimDeployment{}, &CompoundAINimDeploymentList{})
}
