<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribution Guidelines

Contributions that fix documentation errors or that make small changes
to existing code can be contributed directly by following the rules
below and submitting an appropriate PR.

Contributions intended to add significant new functionality must
follow a more collaborative path described in the following
points. Before submitting a large PR that adds a major enhancement or
extension, be sure to submit a GitHub issue that describes the
proposed change so that the Triton team can provide feedback.

- As part of the GitHub issue discussion, a design for your change
  will be agreed upon. An up-front design discussion is required to
  ensure that your enhancement is done in a manner that is consistent
  with Triton Distributed's overall architecture.

- The Triton Distributed project is spread across multiple repos. The Triton team
  will provide guidance about how and where your enhancement should be
  implemented.

- Testing is a critical part of any Triton
  enhancement. You should plan on spending significant time on
  creating tests for your change. The Triton team will help you to
  design your testing so that it is compatible with existing testing
  infrastructure.

- If your enhancement provides a user visible feature then you need to
  provide documentation.

# Contribution Rules

- The code style convention is enforced by clang-format. See below on
  how to ensure your contributions conform. In general please follow
  the existing conventions in the relevant file, submodule, module,
  and project when you add new code or when you extend/fix existing
  functionality.

- Avoid introducing unnecessary complexity into existing code so that
  maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:

  - Avoid committing commented-out code.

  - Wherever possible, each PR should address a single concern. If
    there are several otherwise-unrelated things that should be fixed
    to reach a desired endpoint, it is perfectly fine to open several
    PRs and state in the description which PR depends on another
    PR. The more complex the changes are in a single PR, the more time
    it will take to review those changes.

  - Make sure that the build log is clean, meaning no warnings or
    errors should be present.

  - Make sure all tests pass:

- Triton Distributed's default build assumes recent versions of
  dependencies (CUDA, TensorFlow, PyTorch, TensorRT,
  etc.). Contributions that add compatibility with older versions of
  those dependencies will be considered, but NVIDIA cannot guarantee
  that all possible build configurations work, are not broken by
  future contributions, and retain highest performance.

- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code). You must certify compliance with the [license terms](https://github.com/triton-inference-server/triton-distributed/blob/main/LICENSE) and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org) described below before your pull request (PR) can be merged.

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

# Coding Convention

All pull requests are checked against the
[pre-commit hooks](https://github.com/pre-commit/pre-commit-hooks)
located [in the repository's top-level .pre-commit-config.yaml](https://github.com/NVIDIA/triton-inference-server/blob/master/pre-commit-config.yaml).
The hooks do some sanity checking like linting and formatting.
These checks must pass to merge a change.

To run these locally, you can
[install pre-commit,](https://pre-commit.com/#install)
then run `pre-commit install` inside the cloned repo. When you
commit a change, the pre-commit hooks will run automatically.
If a fix is implemented by a pre-commit hook, adding the file again
and running `git commit` a second time will pass and successfully
commit.
