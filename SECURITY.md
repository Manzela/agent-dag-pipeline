# Security Policy

## Zero Tolerance Protocol

This repository has been sanitized to remove all proprietary identifiers, credentials, and client data. The following classes of information must **never** appear in this codebase:

### Prohibited Content
- Company names, brand names, or client identifiers
- API keys, service account credentials, or tokens
- Internal Jira/project references
- GCP project IDs, bucket names, or resource identifiers
- Any personal identifiable information (PII)

### Automated Enforcement
The CI pipeline includes automated security scanning that checks for:
- Proprietary identifier patterns (grep scan)
- Hardcoded API keys and tokens (regex scan)
- GCP service account key patterns

### Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** create a public GitHub issue
2. Email: security@manzela.dev
3. Include a description of the vulnerability and steps to reproduce
4. Allow 48 hours for initial response

### Fail-Closed Security Model

The pipeline implements fail-closed security at multiple layers:

1. **Store Context Integrity**: Verified at pipeline entry and exit
2. **Immutable Data Contracts**: All Pydantic models use `frozen=True`
3. **Tenant Isolation**: Adapter-level binding prevents cross-contamination
4. **Audit Trail**: Every pipeline exit is recorded with full context
