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

. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/common.ps1"

# == begin common.ps1 extensions ==

$global:_print_template = $null

function default_print_template {
  $value = $false
  write-debug "<default_print_template> -> ${value}."
  return $value
}

function env_get_print_template {
  $value = $($null -ne $env:NVBUILD_PRINT_TEMPLATE)
  write-debug "<env_get_print_template> -> '${value}'."
  return $value
}

function env_set_print_template([bool] $value) {
  if ($null -eq $env:NVBUILD_NOSET) {
    write-debug "<env_set_print_template> value: ${value}."
    if ($value) {
      $env:NVBUILD_PRINT_TEMPLATE = '1'
    }
    else {
      $env:NVBUILD_PRINT_TEMPLATE = $null
    }
  }
}

function get_print_template {
  if ($null -eq $global:_print_template) {
    $value = $(env_get_print_template)
    if ($null -ne $value) {
      set_print_template $value
    }
    else {
      set_print_template $(default_print_template)
    }
  }
  write-debug "<get_print_template> -> ${global:_print_template}."
  return $global:_print_template
}

function set_print_template([bool] $value) {
  write-debug "<set_print_template> value: ${value}."

  $global:_print_template = $value
  env_set_print_template $value
}

# === end common.ps1 extensions ===

function initialize_test([string] $component_kind, [string] $component_name, [string[]]$params, [object[]] $tests) {
  write-debug "<initialize_test> component_kind: ${component_kind}."
  write-debug "<initialize_test> component_name: ${component_name}."
  write-debug "<initialize_test> params.count: $($params.count)."
  write-debug "<initialize_test> tests.count: $($tests.count)."

  $command = $null
  $is_debug = $false
  $is_verbosity_specified = $false
  $test_filter = @()

  if (0 -eq $params.count) {
    write-title './test-chart <command> [<options>]'
    write-high 'commands:'
    write-normal '  list            Prints a list of available tests and quits.'
    write-normal '  test            Executes available tests. (default)'
    write-normal ''
    write-high 'options:'
    write-normal '  --print|-p      Prints the output of the ''helm template'' command to the terminal.'
    write-normal '  -t:<test>       Specifies which tests to run. When not provided all tests will be run.'
    write-normal '                  Use ''list'' to determine which tests are available.'
    write-normal '  -v:<verbosity>  Enables verbose output from the test scripts.'
    write-normal '                  verbosity:'
    write-normal '                    minimal|m:  Sets build-system verbosity to minimal. (default)'
    write-normal '                    normal|n:   Sets build-system verbosity to normal.'
    write-normal '                    detailed|d: Sets build-system verbosity to detailed.'
    write-normal '  --debug         Enables verbose build script tracing; this has no effect on build-system verbosity.'
    write-normal ''
  }

  for ($i = 0 ; $i -lt $params.count ; $i += 1) {
    $arg = $params[$i]
    $arg2 = $null
    $pair = $arg -split ':'

    if ($pair.count -gt 1) {
      $arg = $pair[0]
      if (($null -eq $pair[1]) -and ($pair[1].length -gt 0)) {
        $arg2 = $pair[1]
      }
    }

    if ($i -eq 0) {
      if ('list' -ieq $arg)
      {
        $command = 'LIST'
        continue
      }
      elseif ('test' -ieq $arg) {
        $command = 'TEST'
        continue
      }
      else {
        $command = 'TEST'
      }
    }

    if ('--debug' -ieq $arg) {
      $is_debug = $true
    }
    elseif (('--print' -ieq $arg) -or ('-p' -ieq $arg)) {
      if ('TEST' -ne $command) {
        usage_exit "Option '${arg}' not supported by command 'list'."
      }
      if (get_print_template) {
        usage_exit "Option '${arg}' already specified."
      }
      set_print_template($true)
    }
    elseif (('--test' -ieq $arg) -or ('-t' -ieq $arg))
    {
      if ($null -eq $arg2)
      {
        if ($i + 1 -ge $params.count) {
          usage_exit "Expected value following ""{$arg}""."
        }

        $i += 1
        $test_name = $params[$i]
      }
      else
      {
        $test_name = $arg2
      }

      $test_found = $false

      $parts = $test_name.split('/')
      if ($parts.count -gt 1) {
        $test_name = $parts[$parts.count - 1]
      }

      foreach ($test in $tests) {
        if ($test.name -ieq $test_name) {
          $test_found = $true
          break
        }
      }

      if (-not $test_found) {
        usage_exit "Unknown test name ""${test_name}"" provided."
      }

      $test_filter += $test_name
    }
    elseif (('--verbosity' -ieq $arg) -or ('-v' -ieq $arg)) {
      if ($null -eq $arg2)
      {
        if ($i + 1 -ge $params.count) {
          usage_exit "Expected value following ""{$arg}""."
        }

        $i += 1
        $value = $params[$i]
      }
      else
      {
        $value = $arg2
      }

      if (('minimal' -ieq $value) -or ('m' -ieq $value)) {
        $verbosity = 'MINIMAL'
      }
      elseif (('normal' -ieq $value) -or ('n' -ieq $value)) {
        $verbosity = 'NORMAL'
      }
      elseif (('detailed' -ieq $value) -or ('d' -ieq $value)) {
        $verbosity = 'DETAILED'
      }
      else {
        usage_exit "Invalid verbosity option ""${arg}""."
      }

      $(set_verbosity $verbosity)
      $is_verbosity_specified = $true
    }
    else {
      usage_exit "Unknown option '${arg}'."
    }
  }

  $is_debug = $is_debug -or $(is_debug)
  if ($is_debug) {
    $DebugPreference = 'Continue'
  }
  else {
    $DebugPreference = 'SilentlyContinue'
  }

  # When a subset of tests has been requested, filter out the not requested tests.
  if ($test_filter.count -gt 0) {
    write-debug "<initialize_test> selected.count: $($test_filter.count)."

    $replace = @()

    # Find the test that matches each selected item and add it to a replacement list.
    foreach ($filter in $test_filter) {
      foreach ($test in $tests) {
        if ($test.name -ieq $filter) {
          $replace += $test
          break
        }
      }
    }

    # Replace the test list with the replacement list.
    $tests = $replace
    write-debug "<initialize_test> tests.count = $($tests.count)."
  }

  if ((-not $is_verbosity_specified) -and (-not $(is_tty))) {
    write-debug "<initialize_test> override verbosity with 'detailed' when TTY not detected."
    set_verbosity 'DETAILED'
  }

  return @{
    command = $command
    component = $component_kind
    name = $component_name
    is_debug = $is_debug
    tests = $tests
  }
}

function list_helm_tests([object] $config) {
  write-debug "<list_helm_tests> config.component = '$($config.component)'."
  write-debug "<list_helm_tests> config.name = '$($config.name)'."
  write-debug "<list_helm_tests> config.count = [$($config.tests.count)]"

  if ('LIST' -ne $config.command) {
    throw "List method called when command was 'test'."
  }

  write-title "Available tests:"

  foreach ($test in $config.tests) {
    if ('DETAILED' -eq $(get_verbosity)) {
      write-high "- $($test.name):"

      write-normal '  matches:'
      if ($test.matches.count -gt 0){
        foreach ($match in $test.matches) {
          write-low "    ${match}"
        }
      }
      else {
        write-low '    <none>'
      }
      write-normal '  options:'
      if ($test.options.count -gt 0) {
        foreach ($option in $test.options) {
          write-low "    ${option}"
        }
      }
      else{
        write-low '    <none>'
      }
      write-normal '  values:'
      if ($test.values.count -gt 0) {
        foreach($value in $test.values) {
          write-low "    ${value}"
        }
      }
      else {
        write-low '    <none>'
      }
    }
    else {
      write-minimal "- $($test.name)"
    }
  }

  $(cleanup_after)
}

function test_helm_chart([object] $config) {
  write-debug "<test_helm_chart> config.component = '$($config.component)'."
  write-debug "<test_helm_chart> config.name = '$($config.name)'."
  write-debug "<test_helm_chart> config.count = [$($config.tests.count)]"

  if ('LIST' -eq $config.command) {
    list_helm_tests $config
    return $true
  }

  $timer = [system.diagnostics.stopwatch]::startnew()

  $chart_path = to_local_path "deploy/Kubernetes/$($config.component)/charts/$($config.name)"
  $tests_path = to_local_path "deploy/Kubernetes/$($config.component)/tests/$($config.name)"

  push-location $chart_path

  try {
    $fail_count = 0
    $pass_count = 0
    $total_fail_checks = 0
    $total_pass_checks = 0

    foreach ($test in $config.tests) {
      $fail_checks = 0
      $pass_checks = 0

      $helm_command = "helm template test -f $(resolve-path './values.yaml' -relative)"
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      # First add all values files to the command.
      if (($null -ne $test.values) -and ($test.values.count -gt 0)) {
        foreach ($value in $test.values) {
          write-debug "<test_helm-chart> value = '${value}'."
          $helm_command = "${helm_command} -f $(resolve-path "${tests_path}/${value}" -relative)"
        }
        write-debug "<test_helm_chart> helm_command = '${helm_command}'."
      }

      # Second add all --set options to the command.
      if (($null -ne $test.options) -and $($test.options.count -gt 0)) {
        foreach ($option in $test.options) {
          write-debug "<test_helm_chart> option = '${option}'."
          $helm_command = "${helm_command} --set `"${option}`""
        }
      }

      $helm_command = "${helm_command} . --debug"
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      $captured = invoke-expression "${helm_command} 2>&1" | out-string
      $exit_code = $LASTEXITCODE
      write-debug "<test_helm_chart> expected = $($test.expected)."
      write-debug "<test_helm_chart> actual = ${exit_code}."

      $is_pass = $test.expected -eq $exit_code

      if (-not $is_pass) {
        write-normal ">> Failed: exit code ${exit_code} did not match expected $($test.expected)."  $global:colors.low
        # When the exit code is an unexpected non-zero value, print Helm's output.
        if ($exit_code -ne 0)
        {
          # Disable template printing to avoid a double print.
          set_print_template $false
          write-minimal "Helm Template Output" $global:colors.high
          write-minimal $captured $global:colors.low
        }
      }

      if (($null -ne $test.matches) -and ($test.matches.count -gt 0)) {
        foreach ($match in $test.matches) {
          if ('hashtable' -eq $(typeof $match)) {
            write-debug "<test_helm_chart> match is hashtable"
            write-debug "<test_helm_chart> match.lines.count: $($match.lines.count)."

            # Create a single, large regex from all child elements w/ end of line matches between.
            $alt = ''
            $prefix = "\s{$($match.indent)}"
            foreach ($line in $match.lines) {
              $alt = "${alt}${prefix}${line}\s*[\n\r]{1,2}"
            }
            $regex = $alt
          }
          else {
            $regex = $match
          }

          write-debug "<test_helm_chart> regex = '${regex}'."
          $is_match = $captured -match $regex
          write-debug "<test_helm_chart> is_match = ${is_match}."

          if (-not $is_match) {
            write-normal ">> Failed: output did not match: ""${regex}""." $global:colors.low
          }

          $is_pass = $is_pass -and $is_match
          if ($is_match) {
            $pass_checks += 1
          }
          else {
            $fail_checks += 1
          }
        }
      }

      $total_fail_checks += $fail_checks
      $total_pass_checks += $pass_checks

      if (get_print_template) {
        write-normal "Helm Template Output" $global:colors.high
        write-normal $captured $global:colors.low
      }

      if ($is_pass) {
        $pass_count += 1
        write-passed "$($config.component)/$($config.name)/$($test.name) (passed ${pass_checks} of $($fail_checks + $pass_checks) checks)"
      }
      else {
        $fail_count += 1
        write-failed "$($config.component)/$($config.name)/$($test.name) (failed ${fail_checks} of $($fail_checks + $pass_checks) checks)"
        write-low "  command: ${helm_command}"
      }
    }
  }
  catch {
    pop-location

    throw $_
  }

  pop-location

  $timer.stop()

  if ($fail_count -gt 0) {
    write-minimal "Failed: ${fail_count} [${total_fail_checks}], Passed: ${pass_count} ($total_pass_checks), Tests: $($config.tests.count) [$($total_fail_checks + $total_pass_checks)]" $global:colors.test.failed -no_newline
    write-minimal " ($($timer.elapsed.totalseconds.tostring('0.000')) seconds)" $global:colors.low
    return $false
  }
  else
  {
    write-minimal "Passed: ${pass_count} [${total_pass_checks}], Tests: $($config.tests.count) [$($total_fail_checks + $total_pass_checks)]" $global:colors.test.passed -no_newline
    write-minimal " ($($timer.elapsed.totalseconds.tostring('0.000')) seconds)" $global:colors.low
    return $true
  }

  $(cleanup_after)
}
