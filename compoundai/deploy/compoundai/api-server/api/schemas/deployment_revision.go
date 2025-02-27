package schemas

type DeploymentRevisionSchema struct {
	ResourceSchema
	Creator *UserSchema              `json:"creator"`
	Status  DeploymentRevisionStatus `json:"status" enum:"active,inactive"`
}
