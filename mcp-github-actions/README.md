# GitHub Actions MCP Server

MCP server that provides tools to interact with GitHub Actions - list workflow runs, get job logs, and view error annotations.

## Setup

1. Install dependencies:
```bash
cd mcp-github-actions
pip install -r requirements.txt
```

2. Create a GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` and `workflow` scopes
   - Set as environment variable: `GITHUB_TOKEN=ghp_xxxx`

3. Add to Windsurf MCP config (`~/.codeium/windsurf/mcp_config.json`):
```json
{
  "mcpServers": {
    "github-actions": {
      "command": "python",
      "args": ["c:/llama-cpp-py-sync/mcp-github-actions/server.py"],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

4. Restart Windsurf to load the MCP server.

## Available Tools

- **list_workflow_runs** - List recent workflow runs for a repository
- **get_workflow_run** - Get details of a specific run including all jobs
- **get_job_logs** - Get full logs for a specific job
- **get_workflow_run_annotations** - Get error/warning annotations
- **get_failed_jobs_summary** - Get summary of failed jobs with log tails
