package schemasv1

import "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/modelschemas"

type ClusterMemberSchema struct {
	BaseSchema
	Role    modelschemas.MemberRole `json:"role"`
	Creator *UserSchema             `json:"creator"`
	User    UserSchema              `json:"user"`
	Cluster ClusterSchema           `json:"cluster"`
}
