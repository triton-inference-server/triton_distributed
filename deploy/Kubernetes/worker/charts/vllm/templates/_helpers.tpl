# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Annotation Groups
{{- define "triton.annotations.default" }}
triton-distributed: "{{ .Release.Name }}.{{ .Chart.AppVersion | default "0.0" }}"
{{-   with .Values.kubernetes }}
{{-     with .annotations }}
{{        toYaml . }}
{{-     end }}
{{-   end }}
{{- end -}}

{{- define "triton.annotations.chart" }}
helm.sh/chart: {{ .Chart.Name | quote }}
{{-   template "triton.annotations.default" . }}
{{- end -}}

# Label Groups
{{- define "triton.labels.default" }}
{{-   template "triton.label.appInstance" . }}
{{-   template "triton.label.appName" . }}
{{-   template "triton.label.appPartOf" . }}
{{-   template "triton.label.appVersion" . }}
{{- end -}}

{{- define "triton.labels.chart" }}
{{-   template "triton.labels.default" . }}
{{-   template "triton.label.appManagedBy" . }}
{{-   template "triton.label.chart" . }}
{{-   with .Values.kubernetes }}
{{-     with .labels }}
{{        toYaml . }}
{{-     end }}
{{-   end }}
{{-   template "triton.label.release" . }}
{{- end -}}

# Label Values
{{- define "triton.label.appInstance" }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "triton.label.appManagedBy" }}
{{-   $service_name := "triton-distributed" }}
{{-   with .Release.Service }}
{{-     $service_name = . }}
{{-   end }}
app.kubernetes.io/managed-by: {{ $service_name }}
{{- end }}

{{- define "triton.label.appName" }}
app.kubernetes.io/name: {{ required "Property '.triton.componentName' is required." .Values.triton.componentName }}
{{- end }}

{{- define "triton.label.appPartOf" }}
{{-   $part_of := "triton-distributed" }}
{{-   with .Values.kubernetes }}
{{-     with .partOf }}
{{-       $part_of = . }}
{{-     end }}
{{-   end }}
app.kubernetes.io/part-of: {{ $part_of }}
{{- end }}

{{- define "triton.label.appVersion" }}
app.kubernetes.io/version: {{ .Chart.Version | default "0.0" | quote }}
{{- end }}

{{- define "triton.label.chart" }}
helm.sh/chart: {{ .Chart.Name | quote }}
helm.sh/version: {{ .Chart.Version | default "0.0" | quote }}
{{- end }}

{{- define "triton.label.release" }}
release: "{{ .Chart.Name }}_v{{ .Chart.Version | default "0.0" }}"
{{- end }}
