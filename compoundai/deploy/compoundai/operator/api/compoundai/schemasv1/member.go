package schemasv1

import "github.com/triton-inference-server/triton_distributed/deploy/compoundai/api/compoundai/modelschemas"

type CreateMembersSchema struct {
	Usernames []string                `json:"usernames"`
	Role      modelschemas.MemberRole `json:"role" enum:"guest,developer,admin"`
}

type DeleteMemberSchema struct {
	Username string `json:"username"`
}
