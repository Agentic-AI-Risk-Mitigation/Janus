"""
PDE configuration constants.

Pure data -- no heavy imports (authzed, grpc). Safe to import without
SpiceDB running, which is what most consumers need (e.g. RISK_TO_TAINT).
"""

SCHEMA = """
// --- Identity ---
definition user {}

definition agent {
  relation owner:    user
  relation delegate: user
}

definition role {
  relation member: agent | user
}

// --- Code & Data Resources ---
definition repository {
  relation reader: user | agent | role#member
  relation writer: user | agent | role#member
  relation admin:  user | agent | role#member

  permission read   = reader + writer + admin
  permission write  = writer + admin
  permission manage = admin
}

definition codefile {
  relation repository: repository
  relation viewer: user | agent | role#member
  relation editor: user | agent | role#member

  permission read  = viewer + editor + repository->read
  permission write = editor + repository->write
}

definition environment {
  relation operator: user | agent | role#member
  relation deployer: user | agent | role#member

  permission use    = operator + deployer
  permission deploy = deployer
}

definition secret {
  relation reader: user | agent | role#member
  relation writer: user | agent | role#member

  permission read  = reader + writer
  permission write = writer
}

// --- Tier 0: read only ---
definition tool_view_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_read_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_list_directory {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_grep {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_code_search {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_diff {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_log {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 1: disk writes ---
definition tool_edit_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_create_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_write_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_delete_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 2: local execution & network ---
definition tool_run_tests {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_run_script {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_pip_install {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_fetch_url {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 3: remote git & industry actions ---
definition tool_git_commit {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_push {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_clone {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_query_patient_record {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_fetch_clinical_report {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_update_treatment_plan {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_publish_to_portal {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_send_notification {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 4: dangerous -- no ACL edges ever assigned ---
definition tool_bash_terminal {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_http_request {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_read_secret {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_write_secret {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_deploy {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
"""

TOOL_TAINT_LIMIT: dict[str, int] = {
    "view_file":      90,
    "read_file":      90,
    "list_directory": 90,
    "grep":           90,
    "code_search":    90,
    "git_diff":       90,
    "git_log":        90,
    "edit_file":      70,
    "create_file":    70,
    "write_file":     70,
    "delete_file":    50,
    "run_tests":      40,
    "run_script":     30,
    "pip_install":    30,
    "fetch_url":      10,
    "git_commit":     40,
    "git_push":       20,
    "git_clone":      60,
    "query_patient_record": 90,
    "fetch_clinical_report": 90,
    "update_treatment_plan": 70,
    "publish_to_portal": 20,
    "send_notification": 30,
    "bash_terminal":  10,
    "http_request":   10,
    "read_secret":    10,
    "write_secret":    5,
    "deploy":          5,
}

RISK_TO_TAINT: dict[str, int] = {
    "low":      10,
    "medium":   40,
    "high":     70,
    "critical": 90,
}
