"""Training data for the sensitive value classifier.

Contains curated golden sets of sensitive and safe string values used to train
the SVM-RBF classifier.

Sources: real-world secret patterns (AWS, GitHub, Stripe, JWT, etc.) and
realistic safe strings from production codebases (URLs, config, messages).
"""

# Sensitive values: secrets, keys, credentials, connection strings, paths
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
    # Hex-encoded secrets (mixed case distinguishes from hash digests)
    "a3F7b2C9e1D84f6A0b5C3e7D9f2A1b4C",
    "4B7a9F2c1E8d3A6b5C0f7D2e9A1b8C3d",
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
    "/var/backups/database_credentials.bak",
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
    # Additional passwords (short, high special-char density)
    "Monkey!123$ecure",
    "Qw3rty#2025!Adm",
    "zX9$kM2!pL7@qR4v",
    "S3cur3P@ss!#2024",
    # Heroku / npm / PyPI tokens
    "heroku_api_key=d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a",
    "npm_aBcDeFgHiJkLmNoPqRsTuVwXyZaBcDeFgHiJk",
    "pypi-AgEIcHlwaS5vcmcCJGFiY2RlZi0xMjM0LTU2Nzg",
    # Cloudflare / Datadog
    "cf_api_token_v4_AbCdEfGh1234567890xYz",
    "ddapi_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
    # Sentry DSN (embedded secret in URL)
    "https://abc123def456@o123456.ingest.sentry.io/789012",
    # Hashed passwords (bcrypt, argon2 — appear in config dumps)
    "$2b$12$LJ3m4ks9Xk.W8fGjR5nKxeA1b2C3d4E5f6G7h8I9jKl",
    "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$RdescudvJCsgt3ube",
    # TOTP / MFA secrets
    "JBSWY3DPEHPK3PXPAAAAB3NzaC1yc2EA",
    # Signing / webhook secrets
    "whsec_MbN3oPqRsTuVwXyZ1234567890abcdef",
    # Session tokens
    "eyJzZXNzaW9uSWQiOiIxMjM0NTY3ODkwIiwidXNlciI6ImFkbWluIn0=",
    # Sensitive data paths (broader patterns)
    "/home/user/.config/app/private_key.pem",
    # GitLab tokens
    "glpat-xYzAbCdEfGhIjKlMnOpQrS",
    "glpat-7n2R9kPm4xLq8vBw1cFj5h",
    # Terraform / Vault tokens
    "hvs.CAESIJmN3oPqRsTuVwXyZaBcDeFgHiJkLm",
    "s.5GbF3kLm9NpQ2rV4wX7yZaBcDeFg",
    # DigitalOcean tokens
    "dop_v1_abc123def456789012345678901234567890abcd",
    # Vercel tokens
    "vercel_token_aBcDeFgHiJkLmNoPqRsTuVwXyZ123",
    # Supabase keys
    "sbp_1a2b3c4d5e6f7g8h9i0jklmnopqrst",
    # PEM key variants
    "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEIODg4ODg4A",
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA",
    # Additional connection strings
    "amqp://user:K3yP@ssw0rd@rabbitmq.prod:5672/vhost",
    "smtp://alerts:n0tify!@mail.internal:587",
    # More passwords (varied patterns)
    "Admin!2025@Pr0d",
    "R00t#Access$987",
    "my$ecureP@55word",
    # Firebase / Google service account key fragment
    "AIzaSyDn7B3xPqR9sT2uVwXyZ1aBcDeFgHiJkLm",
    # Anthropic / OpenAI keys
    "sk-ant-api03-aBcDeFgHiJkLmNoPqRsTuVwXyZ",
    "sk-proj-Tn7B3xPqR9sT2uVwXyZ1aBcDeFgH",
    # CircleCI / Travis tokens
    "circle_token_aBcDeFgHiJkLmNoPqRsTuVwX",
    # More webhook / signing secrets
    "whsec_9xK2mP5nR8vL3qW7bJ4cF6hTaBcD",
    "sig_secret_Kj8mNx2Lp3Qr4Vw7Bz9cFh6tA",
    # Additional crypto addresses
    "0x1234567890aBcDeF1234567890AbCdEf12345678",
    "bc1pw508d6qejxtdg4y5r3zarvary0c5xw7kw508d6",
    # Private / sensitive config paths
    "/etc/ssl/private/server.key",
    "C:\\Users\\Admin\\.ssh\\id_ed25519",
    "/home/deploy/secrets.env",
    # Person names with structural name features (PII)
    # Each name has at least one structural signal: particle, initial, or hyphen
    # Names WITHOUT structural features (e.g. "John Doe") are excluded because
    # they are structurally identical to Title Case code labels
    # --- Middle initials / abbreviated titles ---
    "Robert M. Michelson",
    "Elon R. Musk",
    "Dr. Sarah Williams",
    "J. Robert Oppenheimer",
    "C. S. Lewis",
    # --- Hyphenated capital words ---
    "Jean-Pierre Lefebvre",
    "Anne-Marie Slaughter",
    "Karl-Heinz Rummenigge",
    "Mary-Jane Watson",
    "Pierre-Auguste Renoir",
    "Hans-Georg Gadamer",
    "Ernst-Ludwig Kirchner",
    # --- Name particles (lowercase middle words) ---
    "Vincent van Gogh",
    "Leonardo da Vinci",
    "Egbert van der Poel",
    "Ludwig van Beethoven",
    "Oscar de la Renta",
    "Miguel de Cervantes",
    "Charles de Gaulle",
    "Jan van Eyck",
    "Franz von Papen",
    "Diego de la Cruz",
    "Sophie von Kleist",
    "Antonio da Costa",
    "Pieter van Hooch",
    "Simone de Beauvoir",
    "Hernan de Soto",
]

# Safe values: realistic non-secret strings from codebases
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
    # UUIDs (safe identifiers, not secrets)
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "12345678-1234-5678-1234-567812345678",
    "00000000-0000-0000-0000-000000000000",
    # Hash digests used for caching/integrity (not secrets)
    "sha256:5d41402abc4b2a76b9719d911017c592",
    "md5:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    # MIME types
    "multipart/form-data; boundary=----WebKitFormBoundary",
    "application/xml; charset=utf-16",
    # Date/time format strings
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "yyyy-MM-dd HH:mm:ss.SSS",
    # HTTP headers
    "Access-Control-Allow-Origin",
    "X-Forwarded-For",
    # CLI flag names
    "--confidence-threshold",
    "--output-format",
    # License identifiers
    "Apache-2.0",
    "BSD-3-Clause",
    # Semver / version strings
    ">=3.10,<4.0",
    # Docker image references
    "python:3.12-slim-bookworm",
    "ghcr.io/owner/repo:latest",
    # Git refs (public hashes, not secrets)
    "refs/heads/main",
    "origin/feature/add-logging",
    # Logging format strings
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Placeholder / sentinel values
    "__default__",
    # Package names
    "tree-sitter-languages",
    "scikit-learn",
    # Enum-like constant strings
    "pending_review",
    "in_progress",
    # Non-secret base64 (config data, not secrets)
    "aHR0cHM6Ly9leGFtcGxlLmNvbQ==",
    "dXNlci1hZ2VudDogTW96aWxsYQ==",
    # More non-secret hex / hash digests
    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "sha1:da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "blake2b:786a02f742015903c6c6fd852552d272",
    "abc123def456789012345678abcdef01",
    # CSS / display values (hex-heavy but not secrets)
    "#ffffff; background-color: #000000",
    "linear-gradient(90deg, #ff6b6b 0%, #556270 100%)",
    # More multi-word natural language (strengthen word_count signal)
    "Failed to connect to database server",
    "Please enter a valid email address",
    "Successfully processed all pending requests",
    "Unable to parse configuration file",
    "Operation completed with warnings",
    # More CLI / code strings
    "default_output_format",
    "sensitive_value_threshold",
    "max_concurrent_requests",
    # Non-secret lowercase hex (hash digests used for caching/integrity)
    "5d41402abc4b2a76b9719d911017c592",
    "7c211433f02024a5b5903e3fa1d72132",
    "098f6bcd4621d373cade4e832627b4f6",
    # Non-secret base64 (encoded safe config / file names)
    "cHl0aG9uOnRyZWUtc2l0dGVy",
    "Y29kZW4tcmV0cmlldmVy",
    # More lowercase hex hashes (common in caching / checksums)
    "deadbeef0123456789abcdef01234567",
    "c3ab8ff13720e8ad9047dd39466b3c89",
    "b1946ac92492d2347c6235b4d2611184",
    # UI labels with derivational suffixes (-tion, -ment, -ing, -ence)
    # Suffix feature distinguishes these from person names structurally
    "Permission Required",
    "Echo Comments",
    "Tool Settings",
    "Build Configuration",
    "Network Settings",
    "System Preferences",
    "Connection Timeout",
    # CamelCase class/type names (NOT secrets — high entropy, no spaces)
    "HttpClientFactory",
    "AbstractSyntaxTree",
    "EventDispatcher",
    "DatabaseConnection",
    "TokenValidator",
    "GraphQLResolver",
    "Base64Encoder",
    "OAuth2Provider",
    "MD5Hasher",
    "UTF8Decoder",
    # CSS values with hex colors (NOT secrets — hex chars confuse the model)
    "#333333; padding: 10px; margin: 0",
    "border: 1px solid #cccccc",
    "color: #1a1a2e; font-size: 14px",
    "#e0e0e0; border-radius: 4px",
    "background: #f5f5f5; opacity: 0.9",
    "#aabbcc; text-decoration: none",
    # CSS shorthand (no spaces — classifier must not confuse with hex secrets)
    "bg:#666666",
    "bg:#333333",
    "fg:#aabbcc",
    "bg:#1a1a2e bold",
    # ALL CAPS phrases (log headers, section titles — NOT secrets)
    "DEBUG SESSION END",
    "DEBUG SESSION START",
    "FINAL RESPONSE",
    "MODEL RESPONSE",
    "THINKING TRACE",
    "MAX STEPS REACHED",
    "BUILD COMPLETED",
    "TEST PASSED",
    "CONNECTION ESTABLISHED",
    "OPERATION CANCELLED",
    # Indented menu / help text (leading whitespace — NOT secrets)
    "  list              List all items",
    "  show              Show details",
    "  clear             Clear cache",
    "  path              Show config path",
    "     For remote servers:",
    "     For GGUF:",
    "  Select",
    "  Choices:",
    # Compact f-string format specifiers (NOT passwords)
    'f"{value*100:>10.1f}%"',
    'f"{n:04d}"',
    'f"{count:,d} items"',
    'f"{ratio*100:>11.1f}%"',
    # F-string stats lines with ALL-CAPS labels and numbers (NOT secrets)
    'f"  EXACT (100%): {n:3} groups"',
    'f"  TOTAL: {count} items ({pct:.1f}%)"',
    "TEXT-BASED (SEARCH/REPLACE)",
    "AST-BASED (SYMBOL) - for functions",
    "Modification Options (use -R to replace)",
    # CLI help descriptions (short descriptive text)
    "Sensitive Values (-S): minimum confidence",
    "Echo Comments (-E): semantic similarity",
    "Symbol identifier in format module:name",
    "Use --flag to enable this feature",
    "  < 10%  PASS     Low coupling",
    "Detect sensitive values - hardcoded secrets",
    "Flag sensitive values - highlight in code",
    # Title Case label with -tion suffix (structurally distinguishable from names)
    "Flag Action",
    # Lowercase style/error strings (never person names)
    "bold cyan",
    "bold red",
    "no match",
    "no such path",
    "summary",
    "not found",
    "readonly",
    # Strings with formatting chars (colons, pipes — never names)
    "Max steps:",
    "Current:",
    "FastMCP | None",
    "API key override",
    "API endpoint URL",
    "API endpoints",
]
