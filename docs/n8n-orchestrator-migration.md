# n8n Orchestrator Migration Guide

## Context

The BA Agent is a pure SPEC producer — its artifact is `spec_md`. Git operations (branch creation, file commits) and Teamwork mutations (file uploads, stage moves) belong in an orchestrator layer built with n8n.

## Overview

Two n8n workflows needed:
1. **On SPEC Approved** — Triggered when BA Agent marks a session as "approved"
2. **On SPEC Reset** — Triggered when BA Agent resets a session

## Trigger Strategy

**Option A: Webhook from BA Agent** (recommended)
- BA Agent calls n8n webhook URL after marking status
- Immediate, reliable, carries all needed data in payload

**Option B: Database polling**
- n8n polls `ba_sessions` table for status changes
- Simpler but delayed, needs cron schedule

## Auth Headers

**GitHub:**
```
Authorization: Bearer <GITHUB_TOKEN>
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2022-11-28
```

**Teamwork:**
```
Authorization: Basic <base64(TEAMWORK_API_KEY + ":x")>
```

## Naming Conventions

```
Task ID:    "34729"
Title:      "User Export Feature"
Slug:       "34729-user-export-feature"
Branch:     "feature/34729-user-export-feature"
File path:  "34729-user-export-feature/SPEC.md"
File name:  "SPEC-User_Export_Feature.md" (Teamwork attachment)
```

---

## Workflow 1: On SPEC Approved

**Trigger:** Webhook POST from BA Agent

**Webhook payload:**
```json
{
  "session_id": "abc-123",
  "teamwork_task_id": "34729",
  "task_title": "User Export Feature",
  "spec_md": "# SPEC: User Export Feature\n...",
  "project_id": "820301",
  "constitution_url": "https://github.com/owner/repo/blob/main/CONSTITUTION.md"
}
```

### Step 1: Parse repo info from constitution_url

Extract `owner`, `repo`, `base_branch` from the constitution URL using regex:
```
https://github.com/{owner}/{repo}/blob/{branch}/...
```

### Step 2: Get base branch SHA

```bash
curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{base_branch}"
```

Response: `$.object.sha` → `base_sha`

### Step 3: Create feature branch (if not exists)

Check if branch exists:
```bash
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/{owner}/{repo}/git/ref/heads/feature/{slug}"
```

If 404, create it:
```bash
curl -s -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/{owner}/{repo}/git/refs" \
  -d '{
    "ref": "refs/heads/feature/{slug}",
    "sha": "{base_sha}"
  }'
```

### Step 4: Create or update SPEC.md on branch

Check if file exists (to get SHA for update):
```bash
curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/{owner}/{repo}/contents/{slug}/SPEC.md?ref=feature/{slug}"
```

If exists: extract `$.sha` → `file_sha`

Commit file:
```bash
curl -s -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/{owner}/{repo}/contents/{slug}/SPEC.md" \
  -d '{
    "message": "feat(spec): add SPEC.md for {task_title}",
    "content": "{base64_encoded_spec_md}",
    "branch": "feature/{slug}",
    "sha": "{file_sha}"
  }'
```
Note: `sha` field only needed if file already exists (update). Omit for create.

### Step 5: Upload SPEC to Teamwork (presigned URL flow)

**5a.** Get presigned upload URL:
```bash
curl -s \
  -H "Authorization: Basic {tw_auth}" \
  "https://{teamwork_domain}/projects/api/v1/pendingfiles/presignedurl.json?fileName=SPEC-{title_slug}.md&fileSize={byte_length}"
```
Response: `$.ref` → `file_ref`, `$.url` → `upload_url`

**5b.** Upload file to S3:
```bash
curl -s -X PUT \
  -H "Content-Length: {byte_length}" \
  -H "Content-Type: text/markdown" \
  -H "x-amz-acl: public-read" \
  --data-binary "{spec_md_bytes}" \
  "{upload_url}"
```

**5c.** Attach file to task:
```bash
curl -s -X PUT \
  -H "Authorization: Basic {tw_auth}" \
  -H "Content-Type: application/json" \
  "https://{teamwork_domain}/tasks/{teamwork_task_id}.json" \
  -d '{
    "todo-item": {
      "pendingFileAttachments": "{file_ref}"
    }
  }'
```

### Step 6: Move task to "Ready for Design"

```bash
curl -s -X POST \
  -H "Authorization: Basic {tw_auth}" \
  -H "Content-Type: application/json" \
  "https://{teamwork_domain}/projects/api/v3/workflows/12487/stages/64704/tasks.json" \
  -d '{
    "taskIds": [{teamwork_task_id}]
  }'
```

---

## Workflow 2: On SPEC Reset

**Trigger:** Webhook POST from BA Agent

**Webhook payload:**
```json
{
  "teamwork_task_id": "34729",
  "task_title": "User Export Feature",
  "constitution_url": "https://github.com/owner/repo/blob/main/CONSTITUTION.md"
}
```

### Step 1: Delete SPEC.md from GitHub branch

Get file SHA:
```bash
curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/{owner}/{repo}/contents/{slug}/SPEC.md?ref=feature/{slug}"
```

If exists, delete it:
```bash
curl -s -X DELETE \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/{owner}/{repo}/contents/{slug}/SPEC.md" \
  -d '{
    "message": "chore(spec): remove SPEC.md for {task_title}",
    "sha": "{file_sha}",
    "branch": "feature/{slug}"
  }'
```

### Step 2: Delete SPEC file from Teamwork

List files on task:
```bash
curl -s \
  -H "Authorization: Basic {tw_auth}" \
  "https://{teamwork_domain}/tasks/{teamwork_task_id}/files.json"
```

Find file matching `SPEC-{title}*.md`, then delete:
```bash
curl -s -X DELETE \
  -H "Authorization: Basic {tw_auth}" \
  "https://{teamwork_domain}/projects/api/v1/files/{file_id}.json"
```

### Step 3: Move task back to "Ready for Spec"

```bash
curl -s -X POST \
  -H "Authorization: Basic {tw_auth}" \
  -H "Content-Type: application/json" \
  "https://{teamwork_domain}/projects/api/v3/workflows/12487/stages/64711/tasks.json" \
  -d '{
    "taskIds": [{teamwork_task_id}]
  }'
```

---

## Reference: Teamwork Workflow Stages (workflow 12487)

| Stage | ID | Display Order |
|---|---|---|
| To Do | 59825 | 1994 |
| Ready for Spec | 64711 | 1995 |
| Ready for Design | 64704 | 1996 |
| Ready for Development | 64712 | 1996.5 |
| In Progress | 59818 | 1996.75 |
| Ready for Review | 59826 | 1997 |
| Done | 59819 | 2001 |
| DO NOW | 59827 | 2002 |
