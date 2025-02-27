package controller_common

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type updateConditionCustomResource interface {
	client.Object
	StatusConditionsMutator
	StatusConditionsProvider
}

func UpdateCondition(crd updateConditionCustomResource, conditionType string, status metav1.ConditionStatus, reason, message string) bool {
	for i := range crd.GetStatusConditions() {
		if crd.GetStatusConditions()[i].Type == conditionType {
			if crd.GetStatusConditions()[i].Status == status &&
				crd.GetStatusConditions()[i].Reason == reason &&
				crd.GetStatusConditions()[i].Message == message {
				return false // No update needed
			}
			crd.GetStatusConditions()[i].Status = status
			crd.GetStatusConditions()[i].LastTransitionTime = metav1.Now()
			crd.GetStatusConditions()[i].Reason = reason
			crd.GetStatusConditions()[i].Message = message
			// condition updated
			return true
		}
	}
	conditions := append(crd.GetStatusConditions(), metav1.Condition{
		Type:               conditionType,
		Status:             status,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	})
	crd.SetStatusConditions(conditions)
	// condition updated
	return true
}
