# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set-strictmode -version latest

if ($null -eq $(get-command 'git' -ea 0)) {
  throw "Required tool 'git' not found, unable to continue."
}
. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/common.ps1"

$tests = $(get-childitem . -filter 'run.ps1' -recurse)

foreach ($test in $tests) {
  $test = $test
  write-title "Test: ${test}"
  $exit_code = $(run "${test} test -v:detailed" -continue_on_error)
}
