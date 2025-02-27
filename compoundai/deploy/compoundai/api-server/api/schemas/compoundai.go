package schemas

type CompoundAINimApiSchema struct {
	Route  string `json:"route"`
	Doc    string `json:"doc"`
	Input  string `json:"input"`
	Output string `json:"output"`
}

type CompoundAINimManifestSchema struct {
	Service        string                            `json:"service"`
	BentomlVersion string                            `json:"bentoml_version"`
	Apis           map[string]CompoundAINimApiSchema `json:"apis"`
	SizeBytes      uint                              `json:"size_bytes"`
}
