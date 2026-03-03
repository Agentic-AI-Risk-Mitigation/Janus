from janus.policy.enforcer import PolicyEnforcer
from janus.exceptions import PolicyViolation

def test_janus_algorithmic():
    # 1. Initialize Enforcer
    enforcer = PolicyEnforcer()
    
    # 2. Load simple policy (internal format):
    # Allow read_file only for files ending in .txt
    policy = {
        "read_file": [
            (1, 0, {"file_path": {"type": "string", "pattern": ".*\\.txt$"}}, 0)
        ]
    }
    # Load using update because we provide the internal tuple format directly, 
    # load() expects the external json format or shorthand Dict format
    enforcer._policy = policy

    print("--- Testing Janus PolicyEnforcer Logic ---")
    
    # Test 1: Allowed action
    try:
        enforcer.enforce("read_file", {"file_path": "safe_document.txt"})
        print("✅ PASS: 'safe_document.txt' was allowed.")
    except PolicyViolation as e:
        print(f"❌ FAIL: 'safe_document.txt' should be allowed but got {e}")

    # Test 2: Denied action (wrong file extension)
    try:
        enforcer.enforce("read_file", {"file_path": "secret_config.json"})
        print("❌ FAIL: 'secret_config.json' should have been blocked.")
    except PolicyViolation as e:
        print(f"✅ PASS: 'secret_config.json' was blocked as expected. Reason: {e.reason}")

    # Test 3: Denied action (tool not in policy)
    try:
        enforcer.enforce("run_command", {"command": "rm -rf /"})
        print("❌ FAIL: 'run_command' should have been blocked.")
    except PolicyViolation as e:
        print(f"✅ PASS: unregistered tool 'run_command' was blocked. Reason: {e.reason}")

if __name__ == "__main__":
    test_janus_algorithmic()
