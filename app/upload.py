"""Upload plant data straight into the repo from the app.

Streamlit Cloud's filesystem is ephemeral, so uploaded files are committed to
data/raw/ on GitHub via the contents API. A push to data/raw/** triggers the
"Rebuild dashboard data" Action, which reruns the pipeline and commits the
refreshed processed data; Streamlit Cloud then auto-redeploys.

Secrets required (Streamlit -> Settings -> Secrets):

    upload_password = "…"          # gate on this page
    [github]
    token  = "ghp_…"               # PAT with `repo` (contents: read/write)
    repo   = "Thooms-coder/Wastewater-Treatment"
    branch = "main"                # optional, defaults to main
"""

import base64
import os

import requests
import streamlit as st

RAW_DIR = "data/raw"
API = "https://api.github.com"
ALLOWED_EXT = {".csv", ".xlsx"}


def _safe_name(name):
    """Basename only + allowlisted extension — the filename is attacker-controlled
    and lands in a repo path, so never let it escape data/raw/."""
    base = os.path.basename(str(name)).lstrip(".") or "upload"
    ext = os.path.splitext(base)[1].lower()
    if ext not in ALLOWED_EXT:
        return None
    return base


def _cfg():
    try:
        gh = st.secrets.get("github", {})
        return gh.get("token"), gh.get("repo"), gh.get("branch", "main"), st.secrets.get("upload_password")
    except Exception:  # no secrets file configured at all
        return None, None, "main", None


def _commit_file(token, repo, branch, path, content_bytes, message):
    url = f"{API}/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    # Overwriting an existing path requires its current blob sha.
    existing = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)
    sha = existing.json().get("sha") if existing.status_code == 200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()


def render_upload_page(ctx=None):
    st.header("Upload Plant Data")
    token, repo, branch, password = _cfg()

    if not token or not repo:
        st.error(
            "Upload is not configured. Add a `[github]` token/repo (and "
            "`upload_password`) to the app's Streamlit secrets."
        )
        return

    if password and st.text_input("Password", type="password") != password:
        st.info("Enter the upload password to continue.")
        return

    st.caption(
        "Upload gas (H2S / NH3), Water Reclamation, or daily-report files. "
        "They are committed to `data/raw/` and the dashboard rebuilds "
        "automatically in a few minutes."
    )
    files = st.file_uploader("Files", accept_multiple_files=True)

    if files and st.button(f"Commit {len(files)} file(s) to the repo"):
        progress = st.progress(0.0)
        failed = False
        for i, f in enumerate(files, 1):
            name = _safe_name(f.name)
            if name is None:
                st.error(f"✗ {f.name}: only {', '.join(sorted(ALLOWED_EXT))} files are allowed")
                progress.progress(i / len(files))
                continue
            try:
                _commit_file(token, repo, branch, f"{RAW_DIR}/{name}", f.getvalue(), f"Add raw data: {name}")
                st.write(f"✓ {name}")
            except Exception as exc:  # surface the GitHub error to the operator
                failed = True
                st.error(f"✗ {f.name}: {exc}")
            progress.progress(i / len(files))
        if not failed:
            st.success(
                "Uploaded. The dashboard rebuilds via GitHub Actions (~2–4 min), "
                "then Streamlit Cloud redeploys. Reboot the app if it hasn't updated."
            )


if __name__ == "__main__":
    # ponytail: thin API glue; check the encoding and the security-critical name guard.
    blob = b"time,ppm\n1,2\n"
    assert base64.b64decode(base64.b64encode(blob).decode()) == blob
    assert _safe_name("H2S-1.csv") == "H2S-1.csv"
    assert _safe_name("../../.github/workflows/rebuild.yml") is None   # bad ext, and basename'd
    assert _safe_name("/etc/passwd") is None
    assert _safe_name("a/b/c.xlsx") == "c.xlsx"                        # no path escape
    assert _safe_name("evil.py") is None                              # not allowlisted
    print("ok")
