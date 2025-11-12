import os, requests, re, hashlib, json
from fastapi import FastAPI, BackgroundTasks, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, AsyncGenerator
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 全局变量
llm: Tongyi
emb: DashScopeEmbeddings
text_splitter: RecursiveCharacterTextSplitter

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global llm, emb, text_splitter
    print("Pre-loading DashScope (Alibaba Cloud) models...")
    load_dotenv()

    llm = Tongyi(
        model_name=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.2,
    )
    emb = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("Models pre-loaded and ready.")
    yield

app = FastAPI(title="LangChain+Coze Agent Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 环境变量与配置
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "")
ALLOW_EXT = {"kt","java","xml","md","txt","yml","yaml","json","gradle","tex","cls","sty","py"}
CHROMA_DIR = os.path.abspath("./storage/chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

def auth_or_403(token: str):
    if not BACKEND_TOKEN: return
    if token != f"Bearer {BACKEND_TOKEN}" and token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid backend token")

def parse_repo(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)(/.*)?", url.strip())
    if not m: raise ValueError("repo_url 需形如 https://github.com/owner/repo")
    owner, repo, extra = m.group(1), m.group(2), m.group(3) or ""
    subdir = extra.strip("/").split("/",2)[-1] if extra.strip("/") else ""
    return owner, repo, subdir

def get_default_branch(owner, repo, token=None):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept":"application/vnd.github+json"}
    if token: headers["Authorization"]=f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json().get("default_branch", "main")
    except:
        return "main"

def get_tree(owner, repo, branch, token=None):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {"Accept":"application/vnd.github+json"}
    if token: headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "tree" not in data: raise ValueError("获取文件树失败")
        return data.get("tree", [])
    except requests.exceptions.RequestException as e:
        raise ValueError(f"访问 GitHub API 失败 (branch='{branch}'): {e}")

def filter_paths(tree, include_ext, subdir, max_files):
    """
    返回包含 path 与 sha 的条目列表（保留顺序，最多 max_files）。
    每项为一个 dict，与 GitHub tree item 结构兼容（至少包含 'path' 和可能的 'sha'）。
    """
    items=[]
    subdir = subdir.strip("/") if subdir else ""
    for it in tree:
        if it.get("type")!="blob": continue
        p = it.get("path","")
        if subdir and not p.startswith(subdir + "/"): continue
        ext = p.split(".")[-1].lower() if "." in p else ""
        if include_ext and ext not in include_ext: continue
        size = it.get("size", 0)
        # 跳过非常大的文件
        if isinstance(size,int) and size>1_200_000: continue
        items.append({"path": p, "sha": it.get("sha",""), "size": size})
        if len(items)>=max_files: break
    return items

def fetch_raw(owner, repo, branch, path, token=None):
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    headers={}
    if token: headers["Authorization"]=f"Bearer {token}"
    r = requests.get(raw, headers=headers, timeout=30)
    if r.status_code==404: return ""
    r.raise_for_status()
    return r.text

def compute_file_sha256(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def map_chunks_to_line_ranges(raw: str, chunks: List[str]) -> List[Dict[str,int]]:
    """
    对给定原文 raw 与 text_splitter 产生的 chunks，
    计算每个 chunk 在原文中的 start_line 和 end_line（1-based）。
    算法：按序在原文中查找 chunk 的首次出现位置，从上一次位置继续查找以避免多处重复匹配问题。
    返回与 chunks 等长的字典列表：{"start_line":int, "end_line":int}
    如果找不到匹配项，会做 fallback（估算或覆盖整个文件）。
    """
    ranges = []
    pos = 0
    raw_len = len(raw)
    # 为速度，提前计算 prefix newline counts一次性切片时使用 count
    for chunk in chunks:
        if not chunk:
            # 空 chunk（极少见）
            ranges.append({"start_line": 1, "end_line": 1})
            continue
        start_char = raw.find(chunk, pos)
        if start_char == -1:
            # 尝试更宽松的匹配：从 pos 向前回退少量寻找
            # 这里为简洁实现：回退 200 chars 的窗口再查找
            back_pos = max(0, pos - 200)
            start_char = raw.find(chunk, back_pos)
        if start_char == -1:
            # fallback: 预计 chunk 在文件末尾，按剩余字符分配
            # 以上一 chunk 的 end_line + 1 为 start，或 1
            if ranges:
                start_line = ranges[-1]["end_line"] + 1
            else:
                start_line = 1
            end_line = start_line + chunk.count('\n')
            ranges.append({"start_line": start_line, "end_line": end_line})
            # advance pos approximately
            pos = min(raw_len, pos + len(chunk))
            continue
        # 成功定位 start_char
        start_line = raw[:start_char].count('\n') + 1
        end_line = start_line + chunk.count('\n')
        ranges.append({"start_line": start_line, "end_line": end_line})
        pos = start_char + len(chunk)
    return ranges

def get_vectorstore(project_name: str):
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', project_name)
    if not safe_name or not safe_name[0].isalnum(): safe_name="proj_"+safe_name
    if len(safe_name)<3: safe_name=safe_name.ljust(3,'0')
    if len(safe_name)>63: safe_name=safe_name[:63]
    if safe_name.endswith(('_','.')): safe_name=safe_name[:-1]+'0'
    return Chroma(collection_name=safe_name, embedding_function=emb, persist_directory=CHROMA_DIR)

def build_sys_prompt(memo: str):
    """构建用于 LLM 的系统提示词。"""
    base = """你是开发辅助智能体。必须遵守：
- 优先基于检索到的片段与用户上传的文件回答，并在答案末尾列出引用的“文件路径:行号”。
- 当用户提出“修复/修改/重构/补丁”请求，请输出：
  1) 变更计划（条目说明：在哪个文件、哪个方法、做什么改动）
  2) 补丁内容（统一 diff 格式），用 ```diff 包裹
- 信息不足时，明确指出需要补充的文件/模块/路径。
- 回答结构：
  1) 直接答案
  2) 变更计划（如涉及代码修改）
  3) 补丁（如需要）
  4) 引用（文件:行号）
  5) 下一步建议
"""
    if memo.strip():
        base += f"\n（该用户的长期记忆）：\n{memo}\n"
    return base

class ImportRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = ""
    subdir: Optional[str] = ""
    token: Optional[str] = None
    include_ext: Optional[List[str]] = None
    max_files: Optional[int] = 800
    project_name: str

class AskRequest(BaseModel):
    question: str
    project_name: str

def import_repo_task(req: ImportRequest):
    """
    后台导入任务（改动点：保留 chunk 对应的行号与 blob_sha/file_hash）
    """
    print(f"[BG Task] Starting import for project {req.project_name}: {req.repo_url}")
    try:
        owner, repo, subdir0 = parse_repo(req.repo_url)
        token = req.token or GITHUB_TOKEN or None
        branch_to_use = req.branch or get_default_branch(owner, repo, token)
        subdir = req.subdir or subdir0
        include = set(req.include_ext or list(ALLOW_EXT))
        
        print(f"[BG Task] Fetching file tree for {owner}/{repo}@{branch_to_use}...")
        tree = get_tree(owner, repo, branch_to_use, token)
        items = filter_paths(tree, include, subdir, req.max_files or 800)
        print(f"[BG Task] Found {len(items)} files to import.")
        
        docs=[]
        for i, item in enumerate(items):
            p = item.get("path")
            blob_sha = item.get("sha",
            "")
            if i % 20 == 0: print(f"[BG Task] Processing file {i+1}/{len(items)}: {p}")
            try:
                txt = fetch_raw(owner, repo, branch_to_use, p, token)
                if not txt or not txt.strip(): 
                    continue
                # 计算 file hash
                file_hash = compute_file_sha256(txt)
                # 切分成 chunks
                chunks = text_splitter.split_text(txt)
                # 计算每个 chunk 对应的行范围
                ranges = map_chunks_to_line_ranges(txt, chunks)
                for idx, (chunk, rng) in enumerate(zip(chunks, ranges)):
                    meta = {
                        "repo": f"{owner}/{repo}",
                        "branch": branch_to_use,
                        "path": p,
                        "blob_sha": blob_sha,
                        "chunk_index": idx,
                        "start_line": rng.get("start_line", 1),
                        "end_line": rng.get("end_line", rng.get("start_line",1)),
                        "file_hash": file_hash
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))
            except Exception as e:
                print(f"[BG Task] WARNING: failed to process {p}: {e}")
                continue
        
        print(f"[BG Task] Embedding {len(docs)} chunks...")
        vs = get_vectorstore(req.project_name)
        if docs:
            # 可按需分批写入，这里直接 add_documents（在后续改动中应批量/并发）
            vs.add_documents(docs)
        
        print(f"[BG Task] Import completed for project {req.project_name}.")
    except Exception as e:
        print(f"[BG Task] ERROR during import for project {req.project_name}: {e}")

@app.get("/health")
def health(): return {"ok": True}

@app.post("/import")
def import_repo(req: ImportRequest, background_tasks: BackgroundTasks, authorization: Optional[str]=Header(None)):
    auth_or_403(authorization or "")
    background_tasks.add_task(import_repo_task, req)
    return Response(
        status_code=202,
        content='{"message": "Import job started in background. Please wait a moment before asking questions."}',
        media_type="application/json"
    )

@app.post("/ask")
def ask_code(req: AskRequest, authorization: Optional[str]=Header(None)):
    auth_or_403(authorization or "")
    try:
        vs = get_vectorstore(req.project_name)
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        rel_docs = retriever.invoke(req.question)
        citations = []
        if rel_docs:
            for d in rel_docs:
                md = d.metadata or {}
                path = md.get("path","")
                start = md.get("start_line")
                end = md.get("end_line")
                repo = md.get("repo",
                "")
                if start and end:
                    citations.append(f"{repo}/{path}:{start}-{end}")
                else:
                    citations.append(f"{repo}/{path}")
        if not rel_docs:
            snippet = "(未检索到相关片段)"
            citations_text = "(无引用)"
        else:
            citations_text = "\n".join(f"- {c}" for c in citations)
            snippet = "\n\n".join([f"==== {c} ====\n{d.page_content[:800]}" for c, d in zip(citations, rel_docs)])
        
        sys_prompt = build_sys_prompt("")
        messages = [
            {"role":"system","content": sys_prompt},
            {"role":"user","content": f"问题：{req.question}\n\n相关片段（供参考）：\n{snippet}\n\n引用：\n{citations_text}"}
        ]
        resp = llm.invoke(messages)
        return {"answer": resp, "citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"后端处理/ask时异常: {e}")

# ===================== BELOW: Minimal dataset integration & evaluation additions =====================
# 这些代码被追加到文件末尾，实现对本地 Python 数据集的导入、生成补丁与验证功能（不创建新文件夹）
import threading
import uuid
import time
import csv
import subprocess
import sys
from pathlib import Path

# 轻量 JobManager（内存）
class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}  # job_id -> dict(status, created_at, updated_at, logs, meta, result)

    def new_job(self, meta=None):
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "logs": [],
                "meta": meta or {},
                "result": None
            }
        return job_id

    def update(self, job_id, status=None, log_line=None, result=None):
        now = time.time()
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return
            if status:
                j["status"] = status
            if log_line:
                j["logs"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_line}")
            if result is not None:
                j["result"] = result
            j["updated_at"] = now

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self):
        with self._lock:
            return dict(self._jobs)

job_mgr = JobManager()

# 允许的导入扩展（针对 Python 项目）
DEFAULT_ALLOWED_EXT = {"py", "txt", "md", "json", "yaml", "yml"}

class ImportLocalRequest(BaseModel):
    checkout_path: str
    project_name: str
    include_ext: Optional[List[str]] = None
    max_files: Optional[int] = 10000
    batch_size: Optional[int] = 500

class PatchRequest(BaseModel):
    project_name: str
    question: str
    failing_tests: Optional[List[str]] = None

@app.post("/import_local")
def import_local(req: ImportLocalRequest, background_tasks: BackgroundTasks):
    """
    Import a local checkout path into the vectorstore.
    返回 job_id，后台运行。
    """
    if not req.checkout_path or not Path(req.checkout_path).exists():
        raise HTTPException(status_code=400, detail="checkout_path not found")
    meta = {"checkout_path": req.checkout_path, "project_name": req.project_name}
    job_id = job_mgr.new_job(meta=meta)
    background_tasks.add_task(import_local_task, job_id, req)
    return {"job_id": job_id, "status": "queued"}

def import_local_task(job_id: str, req: ImportLocalRequest):
    job_mgr.update(job_id, status="running", log_line="start import_local_task")
    try:
        include = set(req.include_ext or list(DEFAULT_ALLOWED_EXT))
        files = []
        for root, _, filenames in os.walk(req.checkout_path):
            for fn in filenames:
                ext = fn.split(".")[-1].lower() if "." in fn else ""
                if ext not in include:
                    continue
                p = os.path.join(root, fn)
                try:
                    size = os.path.getsize(p)
                    if size > 1_200_000:
                        job_mgr.update(job_id, log_line=f"skip large file {p}")
                        continue
                except:
                    pass
                files.append(p)
                if len(files) >= (req.max_files or 10000):
                    break
            if len(files) >= (req.max_files or 10000):
                break

        job_mgr.update(job_id, log_line=f"Found {len(files)} files")
        vs = get_vectorstore(req.project_name)
        batch = []
        total_chunks = 0
        for i, fpath in enumerate(files):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
            except Exception as e:
                job_mgr.update(job_id, log_line=f"read failed {fpath}: {e}")
                continue
            if not txt.strip():
                continue
            file_hash = compute_file_sha256(txt)
            chunks = text_splitter.split_text(txt)
            for idx, chunk in enumerate(chunks):
                meta = {
                    "path": os.path.relpath(fpath, req.checkout_path),
                    "chunk_index": idx,
                    "file_hash": file_hash
                }
                batch.append(Document(page_content=chunk, metadata=meta))
                total_chunks += 1
                if len(batch) >= (req.batch_size or 500):
                    vs.add_documents(batch)
                    job_mgr.update(job_id, log_line=f"persisted batch {len(batch)}")
                    batch = []
        if batch:
            vs.add_documents(batch)
            job_mgr.update(job_id, log_line=f"persisted final batch {len(batch)}")
        job_mgr.update(job_id, status="done", result={"total_chunks": total_chunks})
    except Exception as e:
        job_mgr.update(job_id, status="error", log_line=f"import error: {e}")

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    j = job_mgr.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_id not found")
    return j

@app.post("/generate_patch")
def generate_patch(req: PatchRequest):
    """
    Ask the agent to generate a patch for project_name using the current vectorstore.
    Returns { patch, raw, citations } where `patch` is the diff text (best-effort extraction).
    """
    if not req.project_name:
        raise HTTPException(status_code=400, detail="project_name required")
    try:
        vs = get_vectorstore(req.project_name)
        # create a retriever: be tolerant to different langchain/chroma APIs
        try:
            retriever = vs.as_retriever(search_kwargs={"k": 6})
            if hasattr(retriever, "get_relevant_documents"):
                rel_docs = retriever.get_relevant_documents(req.question)
            elif hasattr(retriever, "invoke"):
                rel_docs = retriever.invoke(req.question)
            else:
                rel_docs = vs.similarity_search(req.question, k=6)
        except Exception:
            rel_docs = vs.similarity_search(req.question, k=6)
        snippet = ""
        citations = []
        if rel_docs:
            for d in rel_docs:
                md = d.metadata or {}
                path = md.get("path", "")
                citations.append(path)
                snippet += f"==== {path} ====\n{d.page_content[:800]}\n\n"

        sys_prompt = build_sys_prompt("")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content":
             f"问题：{req.question}\n触发测试：{req.failing_tests}\n相关片段：\n{snippet}\n请输出：\n1) 简短修复说明\n2) 统一 diff 补丁（用 ```diff 包裹），不要包含其它多余文本。"}
        ]
        resp = llm.invoke(messages)
        # Try extract ```diff ... ```
        import re
        m = re.search(r"```diff(.*?)```", resp, re.S)
        diff_text = m.group(1).strip() if m else resp
        return {"patch": diff_text, "raw": resp, "citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def apply_patch_and_test(checkout_path: str, patch_text: str, test_cmd: list):
    """
    Apply a patch (git apply --index) in a new branch and run given test command (list).
    Returns (applied_ok: bool, tests_passed: bool, combined_log: str)
    """
    checkout = Path(checkout_path)
    if not checkout.exists():
        return False, False, f"checkout path not found: {checkout_path}"
    try:
        # create/reset branch
        subprocess.run(["git", "-C", checkout_path, "checkout", "-b", "agent-fix"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "-C", checkout_path, "reset", "--hard", "HEAD"], check=False)
        subprocess.run(["git", "-C", checkout_path, "checkout", "-b", "agent-fix"], check=False)
    patch_file = checkout / "agent_patch.diff"
    patch_file.write_text(patch_text, encoding="utf-8")
    res_apply = subprocess.run(["git", "-C", checkout_path, "apply", "--index", str(patch_file)], capture_output=True, text=True)
    applied_ok = (res_apply.returncode == 0)
    test_passed = False
    test_output = ""
    if applied_ok:
        try:
            p = subprocess.run(test_cmd, cwd=checkout_path, capture_output=True, text=True, check=False)
            test_output = p.stdout + "\n" + p.stderr
            test_passed = (p.returncode == 0)
        except Exception as e:
            test_output = str(e)
    combined_log = res_apply.stdout + "\n" + res_apply.stderr + "\n" + test_output
    return applied_ok, test_passed, combined_log

def evaluate_dataset_jsonl(jsonl_path: str, api_base: str = "http://localhost:8000", poll_timeout=1800):
    """
    Simple runner that:
      - reads dataset JSONL (fields: id, checkout_path, test_cmd (list), failing_tests (list))
      - posts /import_local for each sample
      - polls /jobs/{job_id} until done
      - calls /generate_patch
      - applies patch and runs tests locally
      - writes results to results/eval_results.csv
    Run with: python main.py --eval data/bugsinpy_samples.jsonl
    """
    import requests
    p = Path(jsonl_path)
    if not p.exists():
        print("JSONL dataset not found:", jsonl_path)
        return
    out_csv = Path("results_eval.csv")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        sample = json.loads(line)
        sample_id = sample.get("id") or sample.get("project_name") or str(uuid.uuid4())[:8]
        print("Processing sample:", sample_id)
        # 1) import_local
        payload = {
            "checkout_path": sample["checkout_path"],
            "project_name": sample_id
        }
        r = requests.post(api_base + "/import_local", json=payload)
        if r.status_code != 200 and r.status_code != 202:
            print("import_local failed:", r.status_code, r.text)
            rows.append({"id": sample_id, "import_status": "failed"})
            continue
        job_id = r.json().get("job_id")
        # 2) poll job
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            jr = requests.get(api_base + f"/jobs/{job_id}")
            if jr.status_code != 200:
                time.sleep(2); continue
            jdata = jr.json()
            if jdata.get("status") in ("done", "error"):
                break
            time.sleep(2)
        if jdata.get("status") != "done":
            print("import job not done:", jdata.get("status"))
            rows.append({"id": sample_id, "import_status": jdata.get("status")})
            continue
        # 3) generate_patch
        question = sample.get("question") or f"项目有 failing tests: {sample.get('failing_tests')}. 请定位并生成补丁。"
        payload2 = {"project_name": sample_id, "question": question, "failing_tests": sample.get("failing_tests", [])}
        pr = requests.post(api_base + "/generate_patch", json=payload2)
        if pr.status_code != 200:
            print("generate_patch failed:", pr.status_code, pr.text)
            rows.append({"id": sample_id, "import_status": "done", "generate_patch_status": "failed"})
            continue
        patch = pr.json().get("patch", "")
        # 4) apply & test
        test_cmd = sample.get("test_cmd") or sample.get("build_cmd") or ["pytest", "-q"]
        applied_ok, test_passed, log = apply_patch_and_test(sample["checkout_path"], patch, test_cmd)
        rows.append({
            "id": sample_id,
            "import_status": "done",
            "patch_applied": bool(applied_ok),
            "tests_passed": bool(test_passed),
            "apply_log": (log[:8000] if log else ""),
            "raw_patch": (patch[:3000] if patch else "")
        })
        # cleanup branch (best-effort)
        subprocess.run(["git", "-C", sample["checkout_path"], "checkout", "-"], check=False)
        subprocess.run(["git", "-C", sample["checkout_path"], "branch", "-D", "agent-fix"], check=False)
    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("Evaluation finished, results in", out_csv)
    else:
        print("No rows produced.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", help="path to dataset JSONL to run evaluation (each line is a JSON sample)")
    parser.add_argument("--api", default="http://localhost:8000", help="API base for import/generate endpoints")
    args = parser.parse_args()
    if args.eval:
        evaluate_dataset_jsonl(args.eval, api_base=args.api)
    else:
        print("No --eval provided. Start server with uvicorn main:app --reload to use HTTP endpoints.")
