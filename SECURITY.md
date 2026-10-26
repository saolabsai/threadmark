# Security and data handling

Threadmark is a research library, not a hosted service with user isolation or a security guarantee.

## API credentials

Live comparisons read `OPENAI_API_KEY` or ask for a key through a hidden terminal prompt. Keep the key in your environment or a secret manager; do not put it in workflow JSON, source code, command arguments, screenshots, issues, or benchmark artifacts. The library does not need the earlier prototype's Windows credential store.

## Workflow records and generated artifacts

A live comparison sends the workflow catalog, event text, question, and selected evidence to the configured OpenAI endpoint. The updater also receives previous predicted memory. The full-history baseline sends the complete workflow history. Review the provider's current data policies before using confidential records.

The archive and generated run artifacts can contain the original records, predicted facts, questions, responses, and source IDs. Store them with access controls and a retention policy appropriate to your data. Treat run output as sensitive until reviewed. Budget ledgers track accounting and may need to be retained after a failure; they are not public benchmark files.

Records may contain malicious instructions. Structured parsing and field allowlists reduce accidental misuse but do not provide complete prompt-injection protection. Applications should enforce authorization outside model output and should not execute returned text as code or privileged actions.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** option if private reporting is available. Otherwise, contact the maintainers through the [SaoLabs AI organization](https://github.com/saolabsai) and request a private reporting channel. Do not publish keys, private records, or an exploit containing sensitive data in a public issue.

If a credential is exposed, revoke or rotate it with its provider. Removing it from the latest commit does not remove it from Git history or public copies.
