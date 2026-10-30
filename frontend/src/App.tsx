import { ChangeEvent, CSSProperties, DragEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowUpRight, Check, ChevronDown, CircleHelp,
  Clapperboard, Eye, EyeOff, FileVideo2, Fingerprint, Film, FlaskConical,
  Link2, LoaderCircle, LogIn, LogOut, Mail, MapPin, Shield, ShieldAlert,
  Sparkles, Upload, User, X, Zap, Files, FileDown
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AuthState = { access_token: string; refresh_token: string; user_id: string; username: string } | null;

type SuspiciousFrame = {
  frame_index: number; timestamp_sec: number; fake_score: number;
  neighbour_mse: number; neighbour_delta: number; reason: string;
  gradcam_data_url?: string;
  face_data_url?: string;
};

type SuspiciousReport = {
  total_frames: number;
  fps: number;
  suspicious_count: number;
  detector_name: string;
  suspicious_frames?: SuspiciousFrame[];
  evidence_frames?: SuspiciousFrame[];
  frame_scores?: number[];
  note?: string;
};

type Job = {
  id: string; status: "queued" | "running" | "completed" | "failed";
  filename: string; source: string; created_at: string; batch_id?: string;
  metadata: { frame_count: number; fps: number; width: number; height: number; duration_seconds: number; size_bytes: number; sha256: string; md5?: string };
  result?: {
    fake_score: number; real_score: number; frames_analyzed: number;
    frame_scores: number[]; face_frames: string[];
    suspicious?: SuspiciousReport;
    detector_name?: string; assessment_note: string;
  };
  error?: string; video_url: string;
};

type BatchSummary = {
  batch_id: string; total: number; queued: number; running: number;
  completed: number; failed: number; jobs: Job[];
};

type Page = "landing" | "login" | "register" | "reset" | "dashboard";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const allowed = ["mp4", "mov", "avi", "webm", "mpeg", "mpg", "m4v"];
const AUTH_KEY = "dd_auth";

// ---------------------------------------------------------------------------
// Auth storage helpers
// ---------------------------------------------------------------------------

function loadAuth(): AuthState {
  try { return JSON.parse(localStorage.getItem(AUTH_KEY) || "null"); } catch { return null; }
}
function saveAuth(a: AuthState) {
  if (a) localStorage.setItem(AUTH_KEY, JSON.stringify(a));
  else localStorage.removeItem(AUTH_KEY);
}
function authHeaders(auth: AuthState): Record<string, string> {
  return auth ? { Authorization: `Bearer ${auth.access_token}` } : {};
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

export default function App() {
  const [auth, setAuth] = useState<AuthState>(loadAuth);
  const [page, setPage] = useState<Page>("landing");

  const handleAuth = (a: AuthState) => { saveAuth(a); setAuth(a); setPage("dashboard"); };
  const handleLogout = () => { saveAuth(null); setAuth(null); setPage("landing"); };

  if (page === "login") return <LoginPage onAuth={handleAuth} onBack={() => setPage("landing")} onGoRegister={() => setPage("register")} onGoReset={() => setPage("reset")} />;
  if (page === "register") return <RegisterPage onAuth={handleAuth} onBack={() => setPage("login")} />;
  if (page === "reset") return <ResetPage onBack={() => setPage("login")} />;
  if (page === "dashboard") return <Dashboard auth={auth} onLogout={handleLogout} />;
  return <Landing auth={auth} onLogin={() => setPage("login")} onRegister={() => setPage("register")} onDashboard={() => setPage("dashboard")} />;
}

// ---------------------------------------------------------------------------
// Landing
// ---------------------------------------------------------------------------

function Landing({ auth, onLogin, onRegister, onDashboard }: { auth: AuthState; onLogin: () => void; onRegister: () => void; onDashboard: () => void }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="DeepDetect home"><span className="brand-icon"><Shield size={19} strokeWidth={2.2} /></span><span>deepdetect<span className="brand-ai">.ai</span></span></a>
        <nav className="topnav">
          <a href="#how-it-works">How it works</a>
          <a href="#features">Features</a>
        </nav>
        <div className="topbar-actions">
          {auth ? (
            <button className="btn-primary" onClick={onDashboard}><Activity size={15} /> Open workspace</button>
          ) : (
            <>
              <button className="btn-ghost" onClick={onLogin}><LogIn size={15} /> Sign in</button>
              <button className="btn-primary" onClick={onRegister}>Get started</button>
            </>
          )}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> AI-POWERED VIDEO FORENSICS</div>
          <h1>Look closer at the<br /><span>video in front of you.</span></h1>
          <p>Screen videos for deepfake signals, inspect sampled frames with Grad-CAM overlays, detect suspicious timestamps, and export forensic PDF reports.</p>
          <div className="hero-cta">
            <button className="btn-primary btn-lg" onClick={auth ? onDashboard : onRegister}><Activity size={18} /> {auth ? "Open workspace" : "Start for free"} <ArrowUpRight size={16} /></button>
            <div className="hero-note"><CircleHelp size={16} /><span>AI screening can be wrong. Use results as a review cue, never as proof.</span></div>
          </div>
        </section>

        <section className="features-grid" id="features">
          {[
            { icon: <Shield size={22} />, title: "Grad-CAM Explanations", desc: "Saliency maps highlight which regions drove the model decision on each suspicious frame." },
            { icon: <MapPin size={22} />, title: "Suspicious Timestamps", desc: "Automatically flag frames with score spikes, MSE jumps, and neighbour discontinuities." },
            { icon: <Files size={22} />, title: "Batch Uploads", desc: "Analyse up to 10 videos simultaneously. Track progress across the whole batch." },
            { icon: <Fingerprint size={22} />, title: "Forensic PDF Reports", desc: "Evidence-grade PDFs with SHA-256/MD5 hashes, metadata, and Grad-CAM captures." },
            { icon: <FlaskConical size={22} />, title: "Multiple Models", desc: "ResNet-18 baseline, EfficientNet-V2, and MobileNet+XGBoost ensemble available." },
            { icon: <Zap size={22} />, title: "SCRFD / RetinaFace", desc: "State-of-the-art face detection replaces Haar Cascade for sharper face crops." },
          ].map(f => (
            <article key={f.title} className="feature-card">
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </article>
          ))}
        </section>

        <section className="how-section" id="how-it-works">
          <div className="eyebrow">HOW IT WORKS</div>
          <h2>From video to forensic evidence.</h2>
          <div className="how-cards">
            <article><span>01</span><h3>Add a source</h3><p>Upload files, paste a direct URL, or submit a batch. File stays on your machine.</p></article>
            <article><span>02</span><h3>AI screening</h3><p>ResNet-18 + BiLSTM scans faces frame-by-frame, flagging suspicious timestamps with Grad-CAM.</p></article>
            <article><span>03</span><h3>Export evidence</h3><p>Download a PDF report with hashes, metadata, frame scores, and model explanations.</p></article>
          </div>
        </section>
      </main>
      <footer><a className="brand footer-brand" href="#"><span className="brand-icon"><Shield size={16} /></span><span>deepdetect<span className="brand-ai">.ai</span></span></a><span>Video screening for human review</span><span>Local prototype · No authenticity guarantee</span></footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auth pages
// ---------------------------------------------------------------------------

function AuthShell({ children, title, sub }: { children: React.ReactNode; title: string; sub: string }) {
  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-logo"><Shield size={28} strokeWidth={2} /><span>deepdetect<span className="brand-ai">.ai</span></span></div>
        <h2 className="auth-title">{title}</h2>
        <p className="auth-sub">{sub}</p>
        {children}
      </div>
    </div>
  );
}

function LoginPage({ onAuth, onBack, onGoRegister, onGoReset }: { onAuth: (a: AuthState) => void; onBack: () => void; onGoRegister: () => void; onGoReset: () => void }) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const res = await fetch(`${API}/api/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed.");
      onAuth(data);
    } catch (ex) { setErr(ex instanceof Error ? ex.message : "Login failed."); }
    finally { setBusy(false); }
  };

  return (
    <AuthShell title="Welcome back" sub="Sign in to your DeepDetect workspace">
      <form onSubmit={submit} className="auth-form">
        <label>Email<div className="input-wrap"><Mail size={16} /><input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required /></div></label>
        <label>Password<div className="input-wrap"><Shield size={16} /><input type={show ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" required /><button type="button" className="show-pw" onClick={() => setShow(!show)}>{show ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label>
        {err && <div className="auth-err"><ShieldAlert size={15} />{err}</div>}
        <button className="btn-primary btn-full" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={16} />Signing in…</> : <><LogIn size={16} />Sign in</>}</button>
        <div className="auth-links">
          <button type="button" className="link-btn" onClick={onGoReset}>Forgot password?</button>
          <button type="button" className="link-btn" onClick={onGoRegister}>Create account</button>
          <button type="button" className="link-btn" onClick={onBack}>← Back</button>
        </div>
      </form>
    </AuthShell>
  );
}

function RegisterPage({ onAuth, onBack }: { onAuth: (a: AuthState) => void; onBack: () => void }) {
  const [email, setEmail] = useState(""); const [username, setUsername] = useState(""); const [password, setPassword] = useState(""); const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const res = await fetch(`${API}/api/auth/register`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, username, password }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Registration failed.");
      onAuth(data);
    } catch (ex) { setErr(ex instanceof Error ? ex.message : "Registration failed."); }
    finally { setBusy(false); }
  };

  return (
    <AuthShell title="Create account" sub="Start detecting deepfakes in minutes">
      <form onSubmit={submit} className="auth-form">
        <label>Email<div className="input-wrap"><Mail size={16} /><input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required /></div></label>
        <label>Username<div className="input-wrap"><User size={16} /><input type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder="your_name" required minLength={3} /></div></label>
        <label>Password<div className="input-wrap"><Shield size={16} /><input type={show ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="min. 8 characters" required minLength={8} /><button type="button" className="show-pw" onClick={() => setShow(!show)}>{show ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label>
        {err && <div className="auth-err"><ShieldAlert size={15} />{err}</div>}
        <button className="btn-primary btn-full" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={16} />Creating…</> : <>Create account</>}</button>
        <div className="auth-links"><button type="button" className="link-btn" onClick={onBack}>Already have an account?</button></div>
      </form>
    </AuthShell>
  );
}

function ResetPage({ onBack }: { onBack: () => void }) {
  const [email, setEmail] = useState(""); const [token, setToken] = useState(""); const [newPw, setNewPw] = useState("");
  const [step, setStep] = useState<"request" | "confirm" | "done">("request");
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(""); const [devToken, setDevToken] = useState("");

  const requestReset = async (e: FormEvent) => {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const res = await fetch(`${API}/api/auth/password-reset/request`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      setDevToken(data._dev_token || "");
      setStep("confirm");
    } catch (ex) { setErr(ex instanceof Error ? ex.message : "Request failed."); }
    finally { setBusy(false); }
  };

  const confirmReset = async (e: FormEvent) => {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const res = await fetch(`${API}/api/auth/password-reset/confirm`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, new_password: newPw }) });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setStep("done");
    } catch (ex) { setErr(ex instanceof Error ? ex.message : "Reset failed."); }
    finally { setBusy(false); }
  };

  return (
    <AuthShell title="Reset password" sub="We'll send you a reset token">
      {step === "request" && (
        <form onSubmit={requestReset} className="auth-form">
          <label>Email<div className="input-wrap"><Mail size={16} /><input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required /></div></label>
          {err && <div className="auth-err"><ShieldAlert size={15} />{err}</div>}
          <button className="btn-primary btn-full" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={16} />Sending…</> : "Send reset token"}</button>
          <div className="auth-links"><button type="button" className="link-btn" onClick={onBack}>Back to sign in</button></div>
        </form>
      )}
      {step === "confirm" && (
        <form onSubmit={confirmReset} className="auth-form">
          {devToken && <div className="dev-token-note"><b>Dev token:</b> <code>{devToken}</code></div>}
          <label>Reset token<div className="input-wrap"><Shield size={16} /><input type="text" value={token} onChange={e => setToken(e.target.value)} placeholder="Paste token here" required /></div></label>
          <label>New password<div className="input-wrap"><Shield size={16} /><input type="password" value={newPw} onChange={e => setNewPw(e.target.value)} placeholder="min. 8 characters" required minLength={8} /></div></label>
          {err && <div className="auth-err"><ShieldAlert size={15} />{err}</div>}
          <button className="btn-primary btn-full" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={16} />Updating…</> : "Set new password"}</button>
        </form>
      )}
      {step === "done" && (
        <div className="auth-success"><Check size={32} /><strong>Password updated!</strong><button className="btn-primary btn-full" onClick={onBack}>Sign in</button></div>
      )}
    </AuthShell>
  );
}

// ---------------------------------------------------------------------------
// Dashboard (main workspace)
// ---------------------------------------------------------------------------

function Dashboard({ auth, onLogout }: { auth: AuthState; onLogout: () => void }) {
  const [mode, setMode] = useState<"upload" | "link" | "batch">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [url, setUrl] = useState("");
  const [frames, setFrames] = useState(16);
  const [threshold, setThreshold] = useState(0.5);
  const [job, setJob] = useState<Job | null>(null);
  const [batch, setBatch] = useState<BatchSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState<"results" | "suspicious" | "gradcam">("results");
  const inputRef = useRef<HTMLInputElement>(null);
  const batchRef = useRef<HTMLInputElement>(null);

  // Poll single job
  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;
    const t = window.setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/analyses/${job.id}`, { headers: authHeaders(auth) });
        if (!res.ok) {
          let detail = `The analysis service returned HTTP ${res.status}.`;
          try {
            const body = await res.json();
            if (typeof body.detail === "string") detail = body.detail;
          } catch { /* use the HTTP status when the response is not JSON */ }
          setError(detail);
          setJob({ ...job, status: "failed", error: detail });
          return;
        }
        setJob(await res.json());
      } catch {
        const detail = "The analysis service stopped responding. Check the Render API service logs.";
        setError(detail);
        setJob({ ...job, status: "failed", error: detail });
      }
    }, 1800);
    return () => window.clearInterval(t);
  }, [job, auth]);

  useEffect(() => {
    if (job?.status === "failed" && job.error) setError(`Analysis failed: ${job.error}`);
  }, [job]);

  // Poll batch
  useEffect(() => {
    if (!batch) return;
    const done = batch.completed + batch.failed === batch.total;
    if (done) {
      if (!job) {
        const first = batch.jobs.find(j => j.status === "completed");
        if (first) setJob(first);
      }
      return;
    }
    const t = window.setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/analyses/batch/${batch.batch_id}`, { headers: authHeaders(auth) });
        if (res.ok) {
          const updated: BatchSummary = await res.json();
          setBatch(updated);
          if (job) {
            const current = updated.jobs.find(j => j.id === job.id);
            if (current && (current.status !== job.status || current.result !== job.result)) {
              setJob(current);
            }
          }
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => window.clearInterval(t);
  }, [batch, auth, job]);

  const fileMeta = useMemo(() => {
    if (!file) return null;
    return { size: formatBytes(file.size), ext: file.name.split(".").pop()?.toUpperCase() || "VIDEO" };
  }, [file]);

  const setChosenFile = (f?: File) => {
    if (!f) return;
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!ext || !allowed.includes(ext)) { setError(`Supported formats: ${allowed.join(", ")}.`); return; }
    if (f.size > 500 * 1024 * 1024) { setError("Max 500 MB."); return; }
    setError(""); setFile(f); setJob(null);
  };

  const addBatchFiles = (newFiles: File[]) => {
    const valid = newFiles.filter(f => {
      const ext = f.name.split(".").pop()?.toLowerCase();
      return ext && allowed.includes(ext) && f.size <= 500 * 1024 * 1024;
    });
    setBatchFiles(prev => [...prev, ...valid].slice(0, 10));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault(); setError(""); setBusy(true); setJob(null); setBatch(null);
    try {
      if (mode === "upload") {
        if (!file) throw new Error("Choose a video file first.");
        const body = new FormData();
        body.append("file", file);
        const res = await fetch(`${API}/api/analyses/upload?num_frames=${frames}`, { method: "POST", headers: authHeaders(auth), body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Upload failed.");
        setJob(data);
      } else if (mode === "batch") {
        if (batchFiles.length === 0) throw new Error("Add at least one video.");
        const body = new FormData();
        batchFiles.forEach(f => body.append("files", f));
        const res = await fetch(`${API}/api/analyses/batch?num_frames=${frames}`, { method: "POST", headers: authHeaders(auth), body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Batch failed.");
        setBatch(data);
      } else {
        if (!url.trim()) throw new Error("Paste a public video page or direct video link.");
        const res = await fetch(`${API}/api/analyses/url`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders(auth) }, body: JSON.stringify({ url: url.trim(), num_frames: frames }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed.");
        setJob(data);
      }
    } catch (ex) { setError(ex instanceof Error ? ex.message : "Could not start analysis."); }
    finally { setBusy(false); }
  };

  const downloadReport = () => {
    if (!job) return;
    const link = document.createElement("a");
    link.href = `${API}/api/analyses/${job.id}/report`;
    if (auth) link.href += `?token=${auth.access_token}`;
    link.download = `deepdetect-${job.id.slice(0, 10)}.pdf`;
    link.click();
  };

  const score = job?.result?.fake_score ?? 0;
  const flagged = score >= threshold;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="DeepDetect home"><span className="brand-icon"><Shield size={19} strokeWidth={2.2} /></span><span>deepdetect<span className="brand-ai">.ai</span></span></a>
        <nav className="topnav"><a className="nav-current" href="#analyze">Workspace</a><a href="#how-it-works">About</a></nav>
        <div className="topbar-actions">
          {auth && <span className="user-chip"><User size={14} />{auth.username}</span>}
          <button className="btn-ghost" onClick={onLogout}><LogOut size={15} /> Sign out</button>
        </div>
      </header>

      <main id="top">
        <section className="hero hero-compact">
          <div className="eyebrow"><Sparkles size={14} /> VIDEO AUTHENTICITY REVIEW</div>
          <h1>Look closer at the<br /><span>video in front of you.</span></h1>
        </section>

        <section className="workspace-grid" id="analyze">
          <form className="panel intake-panel" onSubmit={submit}>
            <div className="panel-heading">
              <div><div className="step-label">01 <span>·</span> ADD VIDEO</div><h2>Choose your source</h2></div>
              <div className="heading-icon"><FileVideo2 size={19} /></div>
            </div>

            <div className="source-tabs" role="tablist">
              <button type="button" role="tab" aria-selected={mode === "upload"} className={mode === "upload" ? "selected" : ""} onClick={() => { setMode("upload"); setError(""); }}><Upload size={16} /> Upload</button>
              <button type="button" role="tab" aria-selected={mode === "link"} className={mode === "link" ? "selected" : ""} onClick={() => { setMode("link"); setError(""); }}><Link2 size={16} /> URL</button>
              <button type="button" role="tab" aria-selected={mode === "batch"} className={mode === "batch" ? "selected" : ""} onClick={() => { setMode("batch"); setError(""); }}><Files size={16} /> Batch</button>
            </div>

            {mode === "upload" && <>
              <input ref={inputRef} className="hidden-input" type="file" accept={allowed.map(e => `.${e}`).join(",")} onChange={(ev: ChangeEvent<HTMLInputElement>) => setChosenFile(ev.target.files?.[0])} />
              <div className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(ev: DragEvent<HTMLDivElement>) => { ev.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                onDrop={(ev: DragEvent<HTMLDivElement>) => { ev.preventDefault(); setDragging(false); setChosenFile(ev.dataTransfer.files?.[0]); }}
                onClick={() => inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={ev => ev.key === "Enter" && inputRef.current?.click()}>
                <div className="upload-icon">{file ? <Check size={20} /> : <Upload size={20} />}</div>
                {file ? <><strong className="file-name">{file.name}</strong><span>{fileMeta?.ext} · {fileMeta?.size}</span><button type="button" className="remove-file" onClick={ev => { ev.stopPropagation(); setFile(null); }}><X size={15} /></button></> : <><strong>Drop a video here</strong><span>or <em>browse files</em></span></>}
              </div>
              <div className="format-row"><span>MP4</span><span>MOV</span><span>AVI</span><span>WEBM</span><span>+ more</span><small>Up to 500 MB</small></div>
            </>}

            {mode === "link" && <div className="link-box">
              <div className="link-label-row"><label htmlFor="video-url">Video page or direct video URL</label><button type="button" className="paste-button" onClick={async () => { try { setUrl(await navigator.clipboard.readText()); setError(""); } catch { setError("Clipboard access was blocked. Paste the link into the field manually."); } }}>Paste link</button></div>
              <div className="url-input-wrap"><Link2 size={17} /><input id="video-url" type="url" value={url} onChange={ev => setUrl(ev.target.value)} placeholder="https://example.com/video.mp4 or public video page" /></div>
              <div className="link-hint"><Shield size={14} /> Public pages and direct media links are supported when the host exposes a downloadable video.</div>
            </div>}

            {mode === "batch" && <div className="batch-box">
              <input ref={batchRef} className="hidden-input" type="file" multiple accept={allowed.map(e => `.${e}`).join(",")} onChange={(ev: ChangeEvent<HTMLInputElement>) => ev.target.files && addBatchFiles(Array.from(ev.target.files))} />
              <div className="batch-drop" onClick={() => batchRef.current?.click()} role="button" tabIndex={0}>
                <Files size={24} /><strong>Add videos ({batchFiles.length}/10)</strong><span>Click to browse or drop files here</span>
              </div>
              {batchFiles.length > 0 && <ul className="batch-list">
                {batchFiles.map((f, i) => (
                  <li key={i}><span className="batch-fname">{f.name}</span><span>{formatBytes(f.size)}</span><button type="button" onClick={() => setBatchFiles(prev => prev.filter((_, j) => j !== i))}><X size={13} /></button></li>
                ))}
              </ul>}
            </div>}

            <div className="settings-row">
              <div><label htmlFor="sample-depth">Sample depth</label><span>Frames used by the model</span></div>
              <div className="select-wrap"><select id="sample-depth" value={frames} onChange={ev => setFrames(Number(ev.target.value))}>{[8, 16, 24, 32, 40, 48].map(n => <option key={n} value={n}>{n} frames</option>)}</select><ChevronDown size={14} /></div>
            </div>

            {error && <div className="error-message"><ShieldAlert size={16} /> {error}</div>}
            <button className="analyze-button" type="submit" disabled={busy || job?.status === "queued" || job?.status === "running"}>
              {busy ? <><LoaderCircle className="spin" size={17} />Starting…</> : <><Activity size={17} />Analyze video <ArrowUpRight size={16} /></>}
            </button>
            <div className="privacy-line"><Fingerprint size={14} /><span>File hash recorded. Files stay on this machine.</span></div>
          </form>

          <section className="panel result-panel" aria-live="polite">
            <div className="panel-heading">
              <div><div className="step-label">02 <span>·</span> REVIEW</div><h2>Analysis results</h2></div>
              <div className="heading-actions">
                {batch && job && <button className="btn-ghost btn-sm" onClick={() => setJob(null)}><Files size={14} /> Batch list ({batch.total})</button>}
                {job?.status === "completed" && <><span className="complete-pill"><Check size={13} /> Complete</span><button className="btn-ghost btn-sm" onClick={downloadReport}><FileDown size={14} /> PDF</button></>}
              </div>
            </div>

            {/* Batch progress */}
            {batch && !job && <BatchProgress batch={batch} onSelectJob={setJob} auth={auth} threshold={threshold} />}

            {/* Batch switcher bar when a job from batch is selected */}
            {batch && job && (
              <div className="batch-switcher-bar">
                <span className="batch-switcher-title"><Files size={13} /> Batch items:</span>
                <div className="batch-chips-scroll">
                  {batch.jobs.map((bj, i) => (
                    <button
                      key={bj.id}
                      type="button"
                      className={`batch-chip ${job.id === bj.id ? "selected" : ""} ${bj.status}`}
                      onClick={() => setJob(bj)}
                      disabled={bj.status !== "completed"}
                    >
                      <span className="chip-num">#{i + 1}</span>
                      <span className="chip-name">{bj.filename}</span>
                      {bj.result && <b className={bj.result.fake_score >= threshold ? "score-high" : ""}>{Math.round(bj.result.fake_score * 100)}%</b>}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Single job states */}
            {!job && !batch && <div className="empty-result"><div className="empty-art"><div className="art-frame"><Film size={35} /><span className="art-spark s1" /><span className="art-spark s2" /><span className="art-spark s3" /></div></div><strong>Your review will appear here</strong><span>Add a video to see its screening result.</span><div className="empty-checks"><span><Check size={14} /> File fingerprint</span><span><Check size={14} /> Model screening</span><span><Check size={14} /> Suspicious timestamps</span><span><Check size={14} /> Grad-CAM overlays</span></div></div>}
            {job && (job.status === "queued" || job.status === "running") && <div className="processing-state"><div className="processing-orbit"><LoaderCircle className="spin" size={30} /></div><strong>{job.status === "queued" ? "In the analysis queue" : "Reviewing sampled frames"}</strong><span>Results appear when processing finishes.</span><div className="progress-track"><i /></div><small>{job.filename}</small></div>}
            {job?.status === "failed" && <div className="failed-state"><ShieldAlert size={28} /><strong>Analysis couldn't finish</strong><span>{job.error || "Please try again."}</span></div>}

            {job?.status === "completed" && job.result && <>
              {/* Result tabs */}
              <div className="result-tabs">
                <button className={activeTab === "results" ? "selected" : ""} onClick={() => setActiveTab("results")}><Activity size={14} /> Scores</button>
                <button className={activeTab === "suspicious" ? "selected" : ""} onClick={() => setActiveTab("suspicious")}><AlertTriangle size={14} /> Suspicious {(job.result?.suspicious?.suspicious_count ?? 0) > 0 && <span className="badge-count">{job.result?.suspicious?.suspicious_count}</span>}</button>
                <button className={activeTab === "gradcam" ? "selected" : ""} onClick={() => setActiveTab("gradcam")}><Eye size={14} /> Grad-CAM</button>
              </div>

              {activeTab === "results" && <ResultsTab job={job} threshold={threshold} onThresholdChange={setThreshold} flagged={flagged} score={score} auth={auth} />}
              {activeTab === "suspicious" && <SuspiciousTab report={job.result?.suspicious} fps={job.metadata.fps} />}
              {activeTab === "gradcam" && <GradCamTab report={job.result?.suspicious} />}
            </>}
          </section>
        </section>
      </main>
      <footer><a className="brand footer-brand" href="#"><span className="brand-icon"><Shield size={16} /></span><span>deepdetect<span className="brand-ai">.ai</span></span></a><span>Video screening for human review</span><span>Local prototype · No authenticity guarantee</span></footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Result tabs
// ---------------------------------------------------------------------------

function ResultsTab({ job, threshold, onThresholdChange, flagged, score, auth }: { job: Job; threshold: number; onThresholdChange: (v: number) => void; flagged: boolean; score: number; auth: AuthState }) {
  const assessment = flagged ? "Elevated signal" : "No strong signal";
  const result = job.result!;
  const [selectedFrame, setSelectedFrame] = useState(0);
  const frameCount = result.face_frames.length;
  const activeFrame = Math.min(selectedFrame, Math.max(frameCount - 1, 0));
  const activeImage = result.face_frames[activeFrame];
  const activeScore = result.frame_scores[activeFrame] || 0;
  const [videoFailed, setVideoFailed] = useState(false);
  useEffect(() => setSelectedFrame(0), [job.id]);
  useEffect(() => setVideoFailed(false), [job.id]);
  const selectFrame = (index: number) => {
    setSelectedFrame(Math.max(0, Math.min(index, Math.max(frameCount - 1, 0))));
  };
  return (
    <div className="results-content">
      <div className={`verdict-card ${flagged ? "flagged" : "clear"}`}>
        <div className="verdict-icon">{flagged ? <ShieldAlert size={20} /> : <Shield size={20} />}</div>
        <div className="verdict-copy"><span>MODEL SCREENING</span><strong>{assessment}</strong><small>{flagged ? "Above threshold. Human review recommended." : "Below threshold. Does not prove authenticity."}</small></div>
        <div className={`score-ring ${flagged ? "ring-flagged" : "ring-clear"}`} style={{ "--score": `${score * 100}%` } as CSSProperties}><div><strong>{Math.round(score * 100)}</strong><small>%</small></div></div>
      </div>
      <div className="threshold-setting"><div><span>Flag threshold</span><b>{Math.round(threshold * 100)}%</b></div><input aria-label="Flag threshold" type="range" min="0.3" max="0.8" step="0.01" value={threshold} onChange={ev => onThresholdChange(Number(ev.target.value))} /></div>
      {result.detector_name && <div className="detector-badge"><Zap size={13} /><span>{result.detector_name}</span></div>}
      <div className="video-preview">
        {!videoFailed ? <video
          key={job.video_url}
          controls
          preload="metadata"
          playsInline
          src={`${API}${job.video_url}`}
          onError={() => setVideoFailed(true)}
        /> : <img className="video-mjpeg-fallback" src={`${API}/api/analyses/${job.id}/preview`} alt="Video preview fallback" />}
        <div className="preview-overlay"><Clapperboard size={14} /> ORIGINAL VIDEO</div>
        {videoFailed && <div className="video-error-message">The original codec is not supported by this browser. Showing a compatible live preview generated from the analyzed video.</div>}
      </div>
      <div className="metadata-grid">
        <Meta label="Duration" value={formatDuration(job.metadata.duration_seconds)} />
        <Meta label="Resolution" value={`${job.metadata.width} × ${job.metadata.height}`} />
        <Meta label="Frame rate" value={`${job.metadata.fps.toFixed(1)} FPS`} />
        <Meta label="File size" value={formatBytes(job.metadata.size_bytes)} />
      </div>
      <div className="hash-row"><Fingerprint size={16} /><div><span>SHA-256</span><code>{job.metadata.sha256}</code></div><button type="button" onClick={() => navigator.clipboard?.writeText(job.metadata.sha256)}>Copy</button></div>
      <div className="frame-section">
        <div className="section-title"><div><span className="step-label">SAMPLED FRAMES</span><h3>Model score by sample</h3></div><small>{result.frames_analyzed} analyzed</small></div>
        <div className="score-chart"><ScoreChart scores={result.frame_scores} threshold={threshold} /></div>
        {frameCount > 0 ? (
          <div className="frame-browser" aria-label="Sampled frame browser">
            <div className="frame-viewer">
              <button type="button" className="frame-nav" onClick={() => selectFrame(activeFrame - 1)} disabled={activeFrame === 0} aria-label="Previous frame">‹</button>
              <div className="frame-viewer-image">
                <img src={activeImage} alt={`Analyzed frame ${activeFrame + 1}`} />
                <div className="frame-viewer-caption">
                  <span>FRAME {String(activeFrame + 1).padStart(2, "0")} / {frameCount}</span>
                  <b className={activeScore >= threshold ? "score-high" : ""}>{Math.round(activeScore * 100)}% fake signal</b>
                </div>
              </div>
              <button type="button" className="frame-nav" onClick={() => selectFrame(activeFrame + 1)} disabled={activeFrame === frameCount - 1} aria-label="Next frame">›</button>
            </div>
            <div className="frame-strip frame-strip-scroll">
              {result.face_frames.map((img, i) => (
                <button type="button" className={`frame-card ${i === activeFrame ? "active" : ""}`} key={i} onClick={() => selectFrame(i)} aria-label={`View frame ${i + 1}`}>
                  <img src={img} alt={`Frame ${i + 1}`} />
                  <span>F{String(i + 1).padStart(2, "0")}</span>
                  <b className={result.frame_scores[i] >= threshold ? "score-high" : ""}>{Math.round((result.frame_scores[i] || 0) * 100)}%</b>
                </button>
              ))}
            </div>
          </div>
        ) : <div className="sus-empty"><strong>No face frames returned</strong><p>The API completed without sampled face images.</p></div>}
      </div>
      <div className="result-caveat"><CircleHelp size={15} /><span>{result.assessment_note}</span></div>
      <div className="source-line"><span>VIDEO SOURCE</span><strong>{job.source}</strong><small>ID {job.id.slice(0, 10)}</small></div>
    </div>
  );
}

function SuspiciousTab({ report, fps }: { report?: SuspiciousReport; fps: number }) {
  if (!report) return <div className="empty-result"><strong>No suspicious data</strong></div>;
  const suspicious_frames = report.suspicious_frames ?? [];
  const { suspicious_count } = report;
  return (
    <div className="suspicious-panel">
      <div className="suspicious-summary">
        <span className={`sus-badge ${suspicious_count > 0 ? "sus-bad" : "sus-ok"}`}>{suspicious_count > 0 ? <AlertTriangle size={16} /> : <Check size={16} />}{suspicious_count} suspicious frame{suspicious_count !== 1 ? "s" : ""}</span>
        <span className="sus-detector"><Zap size={13} />{report.detector_name}</span>
      </div>
      {suspicious_count === 0 && <div className="sus-empty"><Check size={32} /><strong>No frames flagged</strong><p>{report.note || "All frames were below the detection threshold for score spikes and inter-frame anomalies."}</p></div>}
      {(report.frame_scores?.length ?? 0) > 0 && (
        <div className="sampled-score-list">
          <strong>Sampled frame scores</strong>
          <div className="sampled-score-grid">
            {report.frame_scores!.map((value, index) => (
              <span key={index}><b>F{String(index + 1).padStart(2, "0")}</b><small>{formatTimestamp(index * (report.total_frames / Math.max(report.frame_scores!.length, 1)) / fps)}</small><em>{Math.round(value * 100)}%</em></span>
            ))}
          </div>
        </div>
      )}
      <div className="sus-list">
        {suspicious_frames.map((sf: SuspiciousFrame, i: number) => (
          <div key={i} className="sus-item">
            {sf.face_data_url && <img className="suspect-frame-image" src={sf.face_data_url} alt={`Suspicious sampled frame ${sf.frame_index + 1}`} />}
            <div className="sus-header">
              <span className="sus-time"><MapPin size={13} />{formatTimestamp(sf.timestamp_sec)}</span>
              <span className={`sus-score ${sf.fake_score >= 0.6 ? "high" : "med"}`}>{Math.round(sf.fake_score * 100)}% fake</span>
            </div>
            <div className="sus-reason">{sf.reason}</div>
            <div className="sus-stats">
              <span>Δ MSE: {sf.neighbour_mse.toFixed(0)}</span>
              <span>Score jump: {(sf.neighbour_delta * 100).toFixed(1)}%</span>
              <span>Frame #{sf.frame_index + 1}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GradCamTab({ report }: { report?: SuspiciousReport }) {
  if (!report) return <div className="empty-result"><strong>No Grad-CAM data</strong></div>;
  const frames = (report.suspicious_frames ?? []).filter((sf: SuspiciousFrame) => sf.gradcam_data_url);
  if (frames.length === 0) return (
    <div className="gradcam-fallback">
      <div className="sus-empty"><Eye size={32} /><strong>Visual evidence heatmaps</strong><p>{report.note || "These are the highest-scoring sampled frames. A low-risk video can have evidence frames without suspicious frames."}</p></div>
      <div className="gradcam-grid">
        {(report.evidence_frames ?? report.suspicious_frames ?? []).filter((sf) => sf.gradcam_data_url || sf.face_data_url).map((sf, i) => (
          <div key={i} className="gradcam-item">
            <img src={sf.gradcam_data_url || sf.face_data_url} alt={`Model explanation frame ${sf.frame_index + 1}`} />
            <div className="gradcam-label"><span>{formatTimestamp(sf.timestamp_sec)}</span><strong>{Math.round(sf.fake_score * 100)}% score</strong></div>
          </div>
        ))}
      </div>
    </div>
  );
  return (
    <div className="gradcam-grid">
      {frames.map((sf: SuspiciousFrame, i: number) => (
        <div key={i} className="gradcam-item">
          <img src={sf.gradcam_data_url} alt={`Grad-CAM frame ${sf.frame_index + 1}`} />
          <div className="gradcam-label">
            <span>{formatTimestamp(sf.timestamp_sec)}</span>
            <strong className={sf.fake_score >= 0.6 ? "score-high" : ""}>{Math.round(sf.fake_score * 100)}%</strong>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Batch progress
// ---------------------------------------------------------------------------

function BatchProgress({ batch, onSelectJob, auth, threshold }: { batch: BatchSummary; onSelectJob: (j: Job) => void; auth: AuthState; threshold: number }) {
  const pct = batch.total > 0 ? Math.round(((batch.completed + batch.failed) / batch.total) * 100) : 0;
  return (
    <div className="batch-progress">
      <div className="batch-header"><Files size={20} /><div><strong>Batch analysis</strong><span>{batch.completed + batch.failed}/{batch.total} done · {batch.failed} failed</span></div><span className="batch-pct">{pct}%</span></div>
      <div className="progress-bar"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
      <p className="batch-subhint">Click on any completed video below to open its full forensic report, video player, and Grad-CAM view.</p>
      <div className="batch-jobs">
        {batch.jobs.map(j => (
          <div key={j.id} className={`batch-job-row ${j.status}`} onClick={() => j.status === "completed" && onSelectJob(j)} style={{ cursor: j.status === "completed" ? "pointer" : "default" }}>
            <span className="bj-status">{j.status === "completed" ? <Check size={13} /> : j.status === "failed" ? <X size={13} /> : <LoaderCircle size={13} className="spin" />}</span>
            <span className="bj-name">{j.filename}</span>
            <div className="bj-right">
              {j.result && <span className={`bj-score ${j.result.fake_score >= threshold ? "high" : ""}`}>{Math.round(j.result.fake_score * 100)}%</span>}
              {j.status === "completed" && <span className="bj-inspect-btn">View analysis <ArrowUpRight size={12} /></span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function Meta({ label, value }: { label: string; value: string }) { return <div className="meta-item"><span>{label}</span><strong>{value}</strong></div>; }

function ScoreChart({ scores, threshold }: { scores: number[]; threshold: number }) {
  const points = scores.map((s, i) => `${scores.length < 2 ? 50 : 8 + i * (84 / (scores.length - 1))},${92 - s * 78}`).join(" ");
  const y = 92 - threshold * 78;
  return <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Frame scores"><line x1="5" x2="95" y1={y} y2={y} className="threshold-line" /><polyline points={points} className="score-line" />{scores.map((s, i) => <circle key={i} cx={scores.length < 2 ? 50 : 8 + i * (84 / (scores.length - 1))} cy={92 - s * 78} r="1.4" className={s >= threshold ? "point high" : "point"} />)}</svg>;
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatBytes(b: number) { return b >= 1024 * 1024 ? `${(b / (1024 * 1024)).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`; }
function formatDuration(s: number) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function formatTimestamp(sec: number) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60), ms = Math.round((sec % 1) * 1000);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}
