#!/usr/bin/env python3
"""
MCP Server for GitHub Actions
Provides tools to fetch workflow runs, job logs, and error annotations.
"""

import os
import json
import asyncio
import httpx
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

GITHUB_API = "https://api.github.com"

server = Server("github-actions")


def get_headers() -> dict:
    """Get GitHub API headers with authentication."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def github_request(endpoint: str) -> dict | list | str:
    """Make a request to GitHub API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GITHUB_API}{endpoint}",
            headers=get_headers(),
            timeout=30.0,
        )
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text
        else:
            return {"error": f"HTTP {response.status_code}: {response.text[:500]}"}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_workflow_runs",
            description="List recent workflow runs for a GitHub repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "workflow": {"type": "string", "description": "Workflow filename (e.g., test.yml). Optional."},
                    "status": {"type": "string", "description": "Filter by status: queued, in_progress, completed. Optional."},
                    "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                },
                "required": ["owner", "repo"],
            },
        ),
        Tool(
            name="get_workflow_run",
            description="Get details of a specific workflow run including jobs",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "run_id": {"type": "integer", "description": "Workflow run ID"},
                },
                "required": ["owner", "repo", "run_id"],
            },
        ),
        Tool(
            name="get_job_logs",
            description="Get logs for a specific job in a workflow run",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "job_id": {"type": "integer", "description": "Job ID"},
                },
                "required": ["owner", "repo", "job_id"],
            },
        ),
        Tool(
            name="get_workflow_run_annotations",
            description="Get error/warning annotations for a workflow run (shows test failures, linting errors, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "run_id": {"type": "integer", "description": "Workflow run ID"},
                },
                "required": ["owner", "repo", "run_id"],
            },
        ),
        Tool(
            name="get_failed_jobs_summary",
            description="Get a summary of all failed jobs in a workflow run with their error messages",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "run_id": {"type": "integer", "description": "Workflow run ID"},
                },
                "required": ["owner", "repo", "run_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "list_workflow_runs":
        owner = arguments["owner"]
        repo = arguments["repo"]
        workflow = arguments.get("workflow")
        status = arguments.get("status")
        limit = arguments.get("limit", 10)
        
        if workflow:
            endpoint = f"/repos/{owner}/{repo}/actions/workflows/{workflow}/runs"
        else:
            endpoint = f"/repos/{owner}/{repo}/actions/runs"
        
        params = [f"per_page={limit}"]
        if status:
            params.append(f"status={status}")
        if params:
            endpoint += "?" + "&".join(params)
        
        data = await github_request(endpoint)
        
        if isinstance(data, dict) and "error" in data:
            return [TextContent(type="text", text=json.dumps(data, indent=2))]
        
        runs = data.get("workflow_runs", [])
        summary = []
        for run in runs[:limit]:
            summary.append({
                "id": run["id"],
                "name": run["name"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "branch": run["head_branch"],
                "commit": run["head_sha"][:7],
                "created_at": run["created_at"],
                "url": run["html_url"],
            })
        
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]
    
    elif name == "get_workflow_run":
        owner = arguments["owner"]
        repo = arguments["repo"]
        run_id = arguments["run_id"]
        
        # Get run details
        run_data = await github_request(f"/repos/{owner}/{repo}/actions/runs/{run_id}")
        
        if isinstance(run_data, dict) and "error" in run_data:
            return [TextContent(type="text", text=json.dumps(run_data, indent=2))]
        
        # Get jobs for this run
        jobs_data = await github_request(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs")
        
        jobs_summary = []
        for job in jobs_data.get("jobs", []):
            job_info = {
                "id": job["id"],
                "name": job["name"],
                "status": job["status"],
                "conclusion": job["conclusion"],
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
            }
            # Add failed steps
            failed_steps = [
                {"name": step["name"], "conclusion": step["conclusion"]}
                for step in job.get("steps", [])
                if step.get("conclusion") == "failure"
            ]
            if failed_steps:
                job_info["failed_steps"] = failed_steps
            jobs_summary.append(job_info)
        
        result = {
            "run": {
                "id": run_data["id"],
                "name": run_data["name"],
                "status": run_data["status"],
                "conclusion": run_data["conclusion"],
                "branch": run_data["head_branch"],
                "commit_message": run_data.get("head_commit", {}).get("message", "")[:100],
            },
            "jobs": jobs_summary,
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "get_job_logs":
        owner = arguments["owner"]
        repo = arguments["repo"]
        job_id = arguments["job_id"]
        
        # GitHub returns logs as plain text with redirect
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
                headers=get_headers(),
                timeout=60.0,
            )
            if response.status_code == 200:
                logs = response.text
                # Truncate if too long, but try to capture the end (where errors usually are)
                if len(logs) > 15000:
                    logs = "... [truncated] ...\n\n" + logs[-15000:]
                return [TextContent(type="text", text=logs)]
            else:
                return [TextContent(type="text", text=f"Error fetching logs: HTTP {response.status_code}")]
    
    elif name == "get_workflow_run_annotations":
        owner = arguments["owner"]
        repo = arguments["repo"]
        run_id = arguments["run_id"]
        
        # Get check suites for this commit
        run_data = await github_request(f"/repos/{owner}/{repo}/actions/runs/{run_id}")
        if isinstance(run_data, dict) and "error" in run_data:
            return [TextContent(type="text", text=json.dumps(run_data, indent=2))]
        
        check_suite_id = run_data.get("check_suite_id")
        if not check_suite_id:
            return [TextContent(type="text", text="No check suite found for this run")]
        
        # Get annotations from check runs
        annotations = []
        jobs_data = await github_request(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs")
        
        for job in jobs_data.get("jobs", []):
            check_run_id = job.get("id")
            ann_data = await github_request(
                f"/repos/{owner}/{repo}/check-runs/{check_run_id}/annotations"
            )
            if isinstance(ann_data, list):
                for ann in ann_data:
                    annotations.append({
                        "job": job["name"],
                        "level": ann.get("annotation_level"),
                        "path": ann.get("path"),
                        "line": ann.get("start_line"),
                        "message": ann.get("message"),
                        "title": ann.get("title"),
                    })
        
        if not annotations:
            return [TextContent(type="text", text="No annotations found (errors may be in job logs instead)")]
        
        return [TextContent(type="text", text=json.dumps(annotations, indent=2))]
    
    elif name == "get_failed_jobs_summary":
        owner = arguments["owner"]
        repo = arguments["repo"]
        run_id = arguments["run_id"]
        
        jobs_data = await github_request(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs")
        
        if isinstance(jobs_data, dict) and "error" in jobs_data:
            return [TextContent(type="text", text=json.dumps(jobs_data, indent=2))]
        
        failed_jobs = []
        for job in jobs_data.get("jobs", []):
            if job.get("conclusion") == "failure":
                failed_steps = []
                for step in job.get("steps", []):
                    if step.get("conclusion") == "failure":
                        failed_steps.append(step["name"])
                
                failed_jobs.append({
                    "job_id": job["id"],
                    "job_name": job["name"],
                    "failed_steps": failed_steps,
                })
        
        if not failed_jobs:
            return [TextContent(type="text", text="No failed jobs found in this run")]
        
        # For each failed job, try to get a snippet of the logs
        for fj in failed_jobs:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/actions/jobs/{fj['job_id']}/logs",
                    headers=get_headers(),
                    timeout=60.0,
                )
                if response.status_code == 200:
                    logs = response.text
                    # Extract last 2000 chars which usually contain the error
                    fj["log_tail"] = logs[-2000:] if len(logs) > 2000 else logs
                else:
                    fj["log_tail"] = f"Could not fetch logs: HTTP {response.status_code}"
        
        return [TextContent(type="text", text=json.dumps(failed_jobs, indent=2))]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
