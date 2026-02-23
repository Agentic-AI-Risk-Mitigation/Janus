print("Starting openai import test...", flush=True)
try:
    import openai  # noqa: F401

    print("OpenAI imported successfully.", flush=True)
except ImportError:
    print("OpenAI import failed.", flush=True)
except Exception as e:
    print(f"OpenAI import error: {e}", flush=True)
