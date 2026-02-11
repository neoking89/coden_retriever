"""Training data for the sensitive value classifier.

Contains curated golden sets of sensitive and safe string values used to train
the LogisticRegression classifier. 50 sensitive + 88 safe = 138 total samples.

Sources: real-world secret patterns (AWS, GitHub, Stripe, JWT, etc.) and
realistic safe strings from production codebases (URLs, config, messages).
"""

# 50 sensitive values: secrets, keys, credentials, connection strings, paths
SENSITIVE_VALUES: list[str] = [
    # AWS keys
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    # GitHub tokens
    "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789",
    "github_pat_11AABBC_xYzAbCdEfGhIjKlMnOpQrStUvWxYz",
    # Generic API keys
    "sk-proj-abc123XYZ789defGHI456jklMNO",
    "api_key_9f8e7d6c5b4a3210fedcba98",
    "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    # JWT tokens
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    # Passwords / passphrases
    "P@ssw0rd!2024#Secure",
    "WNW7eewvW67E63&%",
    "x9!mK#2pL$qR8vN",
    "Tr0ub4dor&3horse",
    # Stripe keys
    "sk_live_4eC39HqLyjWDarjtT1zdp7dc",
    "pk_test_TYooMQauvdEDq54NiTphI7jx",
    # Database connection strings
    "postgres://admin:s3cretP@ss@db.prod.internal:5432/users",
    "mysql://root:hunter2@10.0.0.5:3306/customers",
    "mongodb+srv://app:Kj8mNx2Lp@cluster0.abc123.mongodb.net",
    # Private key fragments
    "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC7",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJBALRiMLAH",
    # Hex-encoded secrets
    "a3f7b2c9e1d84f6a0b5c3e7d9f2a1b4c",
    "deadbeef0123456789abcdef01234567",
    # Base64-encoded secrets
    "dGhpcyBpcyBhIHNlY3JldCB0b2tlbg==",
    "c2VjcmV0UGFzc3dvcmQhQDEyMw==",
    # Slack / Discord tokens
    "xoxp-123456789012-123456789012-123456789012-abcdef1234567890abcdef1234567890",
    "MTIzNDU2Nzg5MDEyMzQ1Njc4.GP7hQA.abcdefghijklmnopqrstuvwxyz1234",
    # Azure / GCP keys
    "DefaultEndpointsProtocol=https;AccountName=stor;AccountKey=kX9+bN3zR7pQ2mF5wL8=",
    "AIzaSyA1b2C3d4E5f6G7h8I9jKlMnOpQrStUvWx",
    # SSH keys
    "AAAAB3NzaC1yc2EAAAADAQABAAABgQC7eFz4nP",
    # SendGrid / Twilio
    "SG.abcdefghijklmnop.qrstuvwxyz0123456789ABCDEFGHIJKLMNOPQR",
    "AC1234567890abcdef1234567890abcdef",
    # Sensitive file paths
    "/etc/shadow",
    "/home/john/.ssh/id_rsa",
    "C:\\Users\\Admin\\Documents\\passwords.xlsx",
    "/var/lib/mysql/customer_data.sql",
    "/home/deploy/.aws/credentials",
    "C:\\Users\\John\\AppData\\Local\\Bitcoin\\wallet.dat",
    "/root/.gnupg/secring.gpg",
    "/opt/app/config/secrets.yml",
    "/home/user/tax_returns_2024.pdf",
    "C:\\Recovery\\WindowsRE\\boot.key",
    # Env var style secrets
    "bearer_tkn_8f3a2b1c9d7e6f5a4b3c2d1e",
    "oauth_9xK2mP5nR8vL3qW7bJ4cF6hT",
    # Crypto wallet addresses
    "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD38",
    "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
    # IP + port combos with credentials
    "ftp://backup:G7k!mN2x@192.168.1.50:21",
    "redis://:p4ssW0rd@10.0.0.12:6379/0",
    # Webhook URLs with tokens
    "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
    "https://discord.com/api/webhooks/123456789/aBcDeFgHiJkLmNoPqRsTuVwXyZ",
    # More random high-entropy secrets
    "dGVzdF9zZWNyZXRfZm9yX2VudHJvcHk=",
    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
]

# 94 safe values: realistic non-secret strings from codebases
SAFE_VALUES: list[str] = [
    # URLs and endpoints (no credentials)
    "http://localhost:11434/v1",
    "http://localhost:8080/api/health",
    "https://api.github.com/repos/owner/repo",
    "https://pypi.org/project/model2vec/",
    "wss://stream.example.com/events",
    # Non-sensitive file paths
    "/usr/local/bin/python3",
    "/var/log/application/server.log",
    "C:\\Program Files\\Common Files\\system",
    "src/coden_retriever/models/entities.py",
    "/home/user/.config/coden/settings.json",
    "node_modules/.package-lock.json",
    # Environment variable names (the names, not values)
    "CODEN_RETRIEVER_MODEL_PATH",
    "GIT_TERMINAL_PROMPT",
    "XDG_CONFIG_HOME",
    "PYTHONDONTWRITEBYTECODE",
    # Error and log messages
    "Configuration reset to defaults",
    "No cache for current project",
    "git command not found in PATH",
    "Conversation history cleared",
    "Cache cleared successfully",
    "Debug mode enabled for session",
    "Connection refused: retrying",
    "Unexpected token in JSON at position 42",
    # Config values and format strings
    "vscode://file/{path}:{line}:1",
    "jetbrains://pycharm/navigate/reference",
    "application/json; charset=utf-8",
    "text/html; charset=iso-8859-1",
    # Regex patterns from code
    r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    r"\b(class|function|def)\s+(\w+)",
    r"(?P<name>[^/]+)\.(?P<ext>\w+)$",
    # UUIDs used as request/trace IDs
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    # Hash digests for caching (not secrets)
    "e3b0c44298fc1c149afbf4c8996fb924",
    "d41d8cd98f00b204e9800998ecf8427e",
    # Base64-encoded non-secret config
    "dHJlZS1zaXR0ZXItcHl0aG9u",
    "YXBwbGljYXRpb24vanNvbg==",
    # Version strings and identifiers
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Python/3.12.1 aiohttp/3.9.1",
    # SQL/query fragments (non-sensitive)
    "SELECT name, type FROM entities WHERE active = true",
    "CREATE INDEX idx_entities_name ON entities(name)",
    # Template strings
    "Found {count} entities in {elapsed:.2f}s",
    "Processing file {idx}/{total}: {name}",
    "{entity_type}:{entity_name} at line {line}",
    # Hex color codes and display values
    "#2b2b2b; font-family: monospace",
    "rgba(255, 255, 255, 0.87)",
    # Documentation / help text
    "How do I get started with this project?",
    "Explain the project structure overview",
    # Delimiter-heavy formatting
    "========================================",
    "--- Analysis Complete (3 warnings) ---",
    # Type annotations (string-form forward references in Python code)
    "nx.DiGraph",
    "Optional[str]",
    "dict[str, Any]",
    "list[CodeEntity]",
    "Callable[[int], bool]",
    # Class/variable names used as strings (CamelCase, snake_case)
    "BM25Index",
    "CacheManager",
    "OutputFormat",
    "func_threshold",
    "max_retries",
    "output_path",
    "source_directory",
    # Filenames and extensions
    "bm25_index.pkl",
    "entities.json",
    "settings.yaml",
    "requirements.txt",
    "pyproject.toml",
    "package-lock.json",
    "tsconfig.json",
    # Short dictionary keys / field names
    "arguments",
    "parameters",
    "file_path",
    "confidence",
    "similarity",
    # Field descriptions (Pydantic/argparse help text)
    "Include private functions",
    "Maximum number of results",
    "Exclude test files from analysis",
    "Minimum confidence threshold",
    # Regex patterns (high-entropy but NOT secrets)
    r"^(?P<type>[A-Z][a-zA-Z]*(?:Error|Exception)):\s*(?P<msg>.*)",
    r"(?P<file>[a-zA-Z0-9_\-\.\/\\]+):(?P<line>\d+)",
    r"<{7}\s*SYMBOL\s*\n(.*?)\n={7}",
    r"^(postgres|mysql|mongodb|redis)(\+\w+)?://",
    r"\b(class|function|def|import)\s+(\w+)",
    # Format strings / table templates (common in CLI formatters)
    "f\"{i:<3} | {colored_type} | {metric}\"",
    "f\"{'#':<3} | {'Type':<12} | {'Metric'}\"",
    "f\"{metric:<{_FLAG_METRIC_COL_WIDTH}}\"",
    "f\"{confidence * 100:.0f}%\"",
    "f\"fallback_{uuid.uuid4().hex[:8]}\"",
    # JetBrains/IDE URL templates
    "f\"jetbrains://{ide}/navigate/reference\"",
]
