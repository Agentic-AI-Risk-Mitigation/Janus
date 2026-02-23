import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("DEBUG: Starting individual imports...", flush=True)

try:
    print("DEBUG: Importing dotenv...", flush=True)
    print("DEBUG: dotenv imported.", flush=True)

    print("DEBUG: Importing langchain_openai...", flush=True)
    print("DEBUG: langchain_openai imported.", flush=True)

    print("DEBUG: Importing langchain.agents...", flush=True)
    print("DEBUG: langchain.agents imported.", flush=True)

    print("DEBUG: Importing core.progent_enforcer...", flush=True)
    print("DEBUG: core.progent_enforcer imported.", flush=True)

    print("DEBUG: Importing frameworks.langchain_agent...", flush=True)
    print("DEBUG: frameworks.langchain_agent imported.", flush=True)

except Exception as e:
    print(f"DEBUG: Import failed: {e}", flush=True)
    sys.exit(1)

print("DEBUG: All imports successful.", flush=True)
