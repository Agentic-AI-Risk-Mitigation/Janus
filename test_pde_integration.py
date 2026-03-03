import pytest
from unittest.mock import patch, MagicMock
from janus.agent import JanusAgent
from janus.exceptions import PolicyViolation

# Ensure module is loaded before mock patches it
import janus.policy.pde_enforcer
from authzed.api.v1 import CheckPermissionResponse

@patch('janus.policy.pde_enforcer.GraphInterceptor')
def test_pde_integration_mocked_interceptor(mock_interceptor_class):
    # Setup mock
    mock_interceptor = MagicMock()
    # Assume view_file is allowed, everything else is blocked
    mock_interceptor.check_tool_access.side_effect = lambda tool, role: tool == "view_file"
    mock_interceptor.current_taint_level = 0
    mock_interceptor_class.return_value = mock_interceptor

    # Initialize agent with PDE
    agent = JanusAgent(
        model="openai/gpt-4o",
        api_key="mock-key-for-testing",
        policy_engine="pde",
        agent_role="test_role"
    )

    # Allow tool invocation simulation (enforce)
    # view_file should pass silently
    agent.enforcer.enforce("view_file", {"file_path": "dummy.txt"})

    # run_command should raise PolicyViolation
    with pytest.raises(PolicyViolation) as exc:
        agent.enforcer.enforce("run_command", {"command": "ls"})
    
    assert "Policy-Discovery-Engine Graph" in str(exc.value)


@patch('policy_engine.enforcement.Client')
def test_pde_taint_propagation(mock_client_class):
    """
    Test that updating taint works correctly inside the actual GraphInterceptor
    and that taint monotonically increases.
    """
    mock_client = mock_client_class.return_value
    mock_resp = MagicMock()
    mock_resp.permissionship = CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION
    mock_client.CheckPermission.return_value = mock_resp

    agent = JanusAgent(
        model="openai/gpt-4o",
        api_key="mock-key-for-testing",
        policy_engine="pde",
        agent_role="test_role"
    )

    # Access the real GraphInterceptor inside PDEEnforcer
    interceptor = agent.enforcer.interceptor

    # Initial taint should be 0
    assert interceptor.current_taint_level == 0

    # Agent updates taint to 50
    agent.update_taint(50)
    assert interceptor.current_taint_level == 50

    # Taint strictly increases, updating to 20 should keep it at 50
    agent.update_taint(20)
    assert interceptor.current_taint_level == 50

    # Agent updates taint to 80
    agent.update_taint(80)
    assert interceptor.current_taint_level == 80

    # Trigger a tool access check
    try:
        agent.enforcer.enforce("dummy_tool", {})
    except PolicyViolation:
        pass

    assert mock_client.CheckPermission.called
