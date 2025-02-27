package controller_common

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
)

const (
	// NvidiaAnnotationHashKey indicates annotation name for last applied hash by the operator
	NvidiaAnnotationHashKey = "nvidia.com/last-applied-hash"
)

func SyncResource[T client.Object](ctx context.Context, c client.Client, desired T, namespacedName types.NamespacedName) (T, error) {
	// Retrieve the GroupVersionKind (GVK) of the desired object
	gvk, err := apiutil.GVKForObject(desired, c.Scheme())
	if err != nil {
		return desired, fmt.Errorf("failed to get GVK for object: %w", err)
	}

	// Create a new instance of the object
	obj, err := c.Scheme().New(gvk)
	if err != nil {
		return desired, fmt.Errorf("failed to create a new object for GVK %s: %w", gvk, err)
	}

	// Type assertion to ensure the object implements client.Object
	current, ok := obj.(T)
	if !ok {
		return desired, fmt.Errorf("failed to cast object to the expected type %T", desired)
	}

	// Retrieve the existing resource
	err = c.Get(ctx, namespacedName, current)
	if err != nil {
		if errors.IsNotFound(err) {
			// If the resource doesn't exist, create it
			if err := c.Create(ctx, desired); err != nil {
				return desired, fmt.Errorf("failed to create resource: %w", err)
			}
			return desired, nil
		}
		return desired, fmt.Errorf("failed to get resource: %w", err)
	}

	// Check if the Spec has changed and update if necessary
	if IsSpecChanged(current, desired) {
		if err := c.Update(ctx, desired); err != nil {
			return desired, fmt.Errorf("failed to update resource: %w", err)
		}
	}

	// Return the updated object
	return current, nil
}

// GetResourceHash returns a consistent hash for the given object spec
func GetResourceHash(obj client.Object) string {
	// Convert obj to a map[string]interface{}
	objMap, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}

	var objData map[string]interface{}
	if err := json.Unmarshal(objMap, &objData); err != nil {
		panic(err)
	}

	// Sort keys to ensure consistent serialization
	sortedObjData := SortKeys(objData)

	// Serialize to JSON
	serialized, err := json.Marshal(sortedObjData)
	if err != nil {
		panic(err)
	}

	// Compute the hash
	hasher := sha256.New()
	hasher.Write(serialized)
	return fmt.Sprintf("%x", hasher.Sum(nil))
}

// IsSpecChanged returns true if the spec has changed between the existing one
// and the new resource spec compared by hash.
func IsSpecChanged(current client.Object, desired client.Object) bool {
	if current == nil && desired != nil {
		return true
	}

	hashStr := GetResourceHash(desired)
	foundHashAnnotation := false

	currentAnnotations := current.GetAnnotations()
	desiredAnnotations := desired.GetAnnotations()

	if currentAnnotations == nil {
		currentAnnotations = map[string]string{}
	}
	if desiredAnnotations == nil {
		desiredAnnotations = map[string]string{}
	}

	for annotation, value := range currentAnnotations {
		if annotation == NvidiaAnnotationHashKey {
			if value != hashStr {
				// Update annotation to be added to resource as per new spec and indicate spec update is required
				desiredAnnotations[NvidiaAnnotationHashKey] = hashStr
				desired.SetAnnotations(desiredAnnotations)
				return true
			}
			foundHashAnnotation = true
			break
		}
	}

	if !foundHashAnnotation {
		// Update annotation to be added to resource as per new spec and indicate spec update is required
		desiredAnnotations[NvidiaAnnotationHashKey] = hashStr
		desired.SetAnnotations(desiredAnnotations)
		return true
	}

	return false
}

// SortKeys recursively sorts the keys of a map to ensure consistent serialization
func SortKeys(obj interface{}) interface{} {
	switch obj := obj.(type) {
	case map[string]interface{}:
		sortedMap := make(map[string]interface{})
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			sortedMap[k] = SortKeys(obj[k])
		}
		return sortedMap
	case []interface{}:
		// Check if the slice contains maps and sort them by the "name" field or the first available field
		if len(obj) > 0 {

			if _, ok := obj[0].(map[string]interface{}); ok {
				sort.SliceStable(obj, func(i, j int) bool {
					iMap, iOk := obj[i].(map[string]interface{})
					jMap, jOk := obj[j].(map[string]interface{})
					if iOk && jOk {
						// Try to sort by "name" if present
						iName, iNameOk := iMap["name"].(string)
						jName, jNameOk := jMap["name"].(string)
						if iNameOk && jNameOk {
							return iName < jName
						}

						// If "name" is not available, sort by the first key in each map
						if len(iMap) > 0 && len(jMap) > 0 {
							iFirstKey := firstKey(iMap)
							jFirstKey := firstKey(jMap)
							return iFirstKey < jFirstKey
						}
					}
					// If no valid comparison is possible, maintain the original order
					return false
				})
			}
		}
		for i, v := range obj {
			obj[i] = SortKeys(v)
		}
	}
	return obj
}

// Helper function to get the first key of a map (alphabetically sorted)
func firstKey(m map[string]interface{}) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys[0]
}
