package controller_common

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	CrdRunning      = "running"
	CrdFailed       = "failed"
	CrdSuccessful   = "successful"
	CrdAvailable    = "available"
	CrdPending      = "pending"
	CrdInitializing = "initializing"
)

type SpecProvider interface {
	GetSpec() any
}

type VersionProvider interface {
	GetVersion() string
}

type VersionMutator interface {
	SetVersion(string)
}

type StatusConditionsProvider interface {
	GetStatusConditions() []metav1.Condition
}

type StatusConditionsMutator interface {
	SetStatusConditions(conditions []metav1.Condition)
}

type StateMutator interface {
	SetState(string)
}

type CRD interface {
	client.Object
	SpecProvider
	StatusConditionsProvider
	StatusConditionsMutator
	VersionProvider
	VersionMutator
	StateMutator
}
