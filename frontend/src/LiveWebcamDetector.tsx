import { useEffect, useRef, useState, useCallback, useId } from "react";
import {
  Activity, AlertTriangle, Camera, Check, ChevronDown, CircleHelp,
  Fingerprint, FlaskConical, Play, Radio, RefreshCw, Shield, ShieldAlert,
  Sparkles, Square, Video, VideoOff, Zap
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type LiveBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
};

export type LiveResult = {
  type: string;
  face_detected: boolean;
  box: LiveBox | null;
  all_faces: LiveBox[];
  fake_score: number;
  real_score: number;
  confidence: number;
  verdict: string;
  risk_level: "LOW" | "SUSPICIOUS" | "HIGH" | "WAITING";
  risk_label: string;
  buffer_count: number;
  buffer_target: number;
  is_buffer_full: boolean;
  frame_scores: number[];
  face_crop_data_url: string | null;
  recent_crops: string[];
  detect_ms: number;
  inference_ms: number;
  fps: number;
  model_name: string;
  detector_name: string;
  frame_width: number;
  frame_height: number;
  timestamp: number;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getWsUrl(): string {
  const envApi = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
  if (envApi.startsWith("http://")) {
    return envApi.replace(/^http:/, "ws:") + "/api/ws/live";
  }
  if (envApi.startsWith("https://")) {
    return envApi.replace(/^https:/, "wss:") + "/api/ws/live";
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const hostname = window.location.hostname || "127.0.0.1";
  const port = window.location.port;

  // In development (e.g. Vite running on 5173, 5174, etc.), connect directly to FastAPI backend on 8000
  if (port && port !== "8000") {
    return `${protocol}//${hostname}:8000/api/ws/live`;
  }
  return `${protocol}//${window.location.host}/api/ws/live`;
}

// ---------------------------------------------------------------------------
// Component: LiveWebcamDetector
// ---------------------------------------------------------------------------

export default function LiveWebcamDetector({
  threshold,
  onThresholdChange,
}: {
  threshold: number;
  onThresholdChange: (t: number) => void;
}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [mirrored, setMirrored] = useState(true);
  const [audioAlert, setAudioAlert] = useState(false);

  // Live telemetry state
  const [latestResult, setLatestResult] = useState<LiveResult | null>(null);
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  const [faceCount, setFaceCount] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isSendingRef = useRef(false);
  const lastSendTimeRef = useRef(0);
  const frameIntervalRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const liveSelectId = useId();

  // Load available video input devices
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === "videoinput");
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch {
      // Ignore if device enumeration is blocked
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // Audio alert on high risk
  const triggerAlertSound = useCallback(() => {
    if (!audioAlert) return;
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      }
      const ctx = audioContextRef.current;
      if (ctx.state === "suspended") ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.15);
    } catch {
      // Audio autoplay policy handled silently
    }
  }, [audioAlert]);

  // Draw face bounding boxes & HUD on overlay canvas
  const drawOverlay = useCallback(
    (res: LiveResult) => {
      const canvas = overlayCanvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, width, height);

      if (!res.face_detected || !res.box) {
        return;
      }

      const frameW = res.frame_width || width;
      const frameH = res.frame_height || height;
      const scaleX = width / frameW;
      const scaleY = height / frameH;

      const isHigh = res.fake_score >= threshold;
      const isSuspicious = !isHigh && res.fake_score >= 0.4;
      const color = isHigh ? "#ef4444" : isSuspicious ? "#f59e0b" : "#22c55e";

      // Draw secondary faces if any
      if (res.all_faces && res.all_faces.length > 1) {
        ctx.strokeStyle = "rgba(180, 190, 200, 0.4)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        for (let i = 1; i < res.all_faces.length; i++) {
          const fb = res.all_faces[i];
          const bx = mirrored ? width - (fb.x + fb.w) * scaleX : fb.x * scaleX;
          const by = fb.y * scaleY;
          const bw = fb.w * scaleX;
          const bh = fb.h * scaleY;
          ctx.strokeRect(bx, by, bw, bh);
        }
        ctx.setLineDash([]);
      }

      // Draw primary face box
      const pb = res.box;
      const bx = mirrored ? width - (pb.x + pb.w) * scaleX : pb.x * scaleX;
      const by = pb.y * scaleY;
      const bw = pb.w * scaleX;
      const bh = pb.h * scaleY;

      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;

      // Corner bracket design for forensic reticle aesthetic
      const corner = Math.min(bw, bh) * 0.22;
      ctx.beginPath();
      // Top-left
      ctx.moveTo(bx, by + corner);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + corner, by);
      // Top-right
      ctx.moveTo(bx + bw - corner, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + corner);
      // Bottom-right
      ctx.moveTo(bx + bw, by + bh - corner);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw - corner, by + bh);
      // Bottom-left
      ctx.moveTo(bx + corner, by + bh);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx, by + bh - corner);
      ctx.stroke();

      // Thin inner boundary
      ctx.shadowBlur = 0;
      ctx.strokeStyle = `${color}44`;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx, by, bw, bh);

      // Top badge: Face Detection
      const topText = `SCRFD: ${Math.round(pb.confidence * 100)}%`;
      ctx.font = "bold 11px 'DM Sans', sans-serif";
      const topMetrics = ctx.measureText(topText);
      const topBadgeW = topMetrics.width + 12;
      const topBadgeH = 18;

      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.fillRect(bx, Math.max(0, by - topBadgeH - 4), topBadgeW, topBadgeH);
      ctx.strokeStyle = color;
      ctx.strokeRect(bx, Math.max(0, by - topBadgeH - 4), topBadgeW, topBadgeH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(topText, bx + 6, Math.max(0, by - topBadgeH - 4) + 13);

      // Bottom badge: Real-time Fake Risk Score
      const botText = `RISK: ${Math.round(res.fake_score * 100)}% · ${res.verdict}`;
      const botMetrics = ctx.measureText(botText);
      const botBadgeW = botMetrics.width + 12;
      const botBadgeH = 20;

      ctx.fillStyle = isHigh ? "rgba(220, 38, 38, 0.9)" : "rgba(15, 23, 42, 0.85)";
      ctx.fillRect(bx, by + bh + 4, botBadgeW, botBadgeH);
      ctx.strokeStyle = color;
      ctx.strokeRect(bx, by + bh + 4, botBadgeW, botBadgeH);
      ctx.fillStyle = isHigh ? "#ffffff" : color;
      ctx.fillText(botText, bx + 6, by + bh + 4 + 14);
    },
    [mirrored, threshold]
  );

  // Initialize and connect WebSocket
  const connectWebSocket = useCallback(() => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const wsUrl = getWsUrl();
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      setCameraError(null);
      isSendingRef.current = false;
      try {
        ws.send(JSON.stringify({ type: "ping" }));
      } catch {}
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "result") {
          const res = data as LiveResult;
          setLatestResult(res);
          setFrameCount((prev) => prev + 1);
          if (res.face_detected) {
            setFaceCount((prev) => prev + 1);
          }
          setScoreHistory((prev) => [...prev.slice(-39), res.fake_score]);
          drawOverlay(res);
          if (res.fake_score >= threshold) {
            triggerAlertSound();
          }
        }
      } catch (err) {
        console.error("WebSocket message parse error:", err);
      } finally {
        isSendingRef.current = false;
      }
    };

    ws.onerror = (e) => {
      console.warn("WebSocket live error:", e);
      setWsConnected(false);
      isSendingRef.current = false;
    };

    ws.onclose = () => {
      setWsConnected(false);
      isSendingRef.current = false;
      // Auto-reconnect if stream is active
      if (streamRef.current && streamRef.current.active) {
        setTimeout(() => {
          if (streamRef.current && streamRef.current.active) {
            connectWebSocket();
          }
        }, 1500);
      }
    };
  }, [drawOverlay, threshold, triggerAlertSound]);

  // Start webcam feed and streaming loop
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Webcam access is not supported by your browser environment.");
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? {
              deviceId: { exact: selectedDeviceId },
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 15, max: 24 },
            }
          : {
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 15, max: 24 },
            },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Prepare offscreen canvas for frame extraction
      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement("canvas");
      }
      offscreenCanvasRef.current.width = 640;
      offscreenCanvasRef.current.height = 480;

      connectWebSocket();
      setIsStreaming(true);

      // Start lock-step frame capture loop (~15 FPS, every 66ms)
      if (frameIntervalRef.current) {
        window.clearInterval(frameIntervalRef.current);
      }

      frameIntervalRef.current = window.setInterval(() => {
        captureAndSendFrame();
      }, 66);
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Could not access the laptop webcam. Check browser permissions.";
      setCameraError(msg);
      setIsStreaming(false);
    }
  };

  // Stop webcam feed and clean up resources
  const stopCamera = () => {
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }

    setIsStreaming(false);
    setWsConnected(false);
    isSendingRef.current = false;
  };

  // Reset 32-frame sequence session
  const resetSession = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
    setScoreHistory([]);
    setFrameCount(0);
    setFaceCount(0);
    setLatestResult((prev) =>
      prev
        ? {
            ...prev,
            buffer_count: 0,
            is_buffer_full: false,
            fake_score: 0.0,
            real_score: 1.0,
            verdict: "REAL",
            risk_level: "LOW",
            recent_crops: [],
          }
        : null
    );
  };

  // Capture current video frame and transmit via WebSocket
  const captureAndSendFrame = () => {
    const video = videoRef.current;
    const canvas = offscreenCanvasRef.current;
    const ws = wsRef.current;

    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const now = Date.now();
    // Safety watchdog: unlock if previous frame has been pending for over 1000ms
    if (isSendingRef.current && now - lastSendTimeRef.current > 1000) {
      isSendingRef.current = false;
    }

    // Skip if previous frame is still being processed by the server (lock-step)
    if (isSendingRef.current) {
      return;
    }

    if (video.readyState < 2 || video.paused || video.ended) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.drawImage(video, 0, 0, w, h);

    try {
      const dataUrl = canvas.toDataURL("image/jpeg", 0.72);
      isSendingRef.current = true;
      lastSendTimeRef.current = now;
      ws.send(JSON.stringify({ type: "frame", image: dataUrl }));
    } catch {
      isSendingRef.current = false;
    }
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const fakeScore = latestResult?.fake_score ?? 0;
  const isHighRisk = fakeScore >= threshold;
  const isSuspicious = !isHighRisk && fakeScore >= 0.4;
  const bufferCount = latestResult?.buffer_count ?? 0;
  const bufferTarget = latestResult?.buffer_target ?? 32;

  return (
    <div className="live-webcam-workspace">
      {/* Top Banner / Controls Bar */}
      <div className="live-controls-bar">
        <div className="live-status-pill-group">
          <span
            className={`live-indicator-pill ${
              isStreaming ? (wsConnected ? "active" : "connecting") : "idle"
            }`}
          >
            <Radio size={14} className={isStreaming ? "pulse" : ""} />
            {isStreaming
              ? wsConnected
                ? "LIVE WEBCAM STREAM"
                : "CONNECTING WEBSOCKET…"
              : "WEBCAM STANDBY"}
          </span>

          <span className="live-subpill">
            <Zap size={12} />
            SCRFD + MobileNetV3/XGBoost
          </span>

          <span className="live-subpill">
            <Activity size={12} />
            {latestResult?.fps ? `${latestResult.fps.toFixed(1)} FPS` : "0.0 FPS"}
          </span>
        </div>

        <div className="live-actions">
          {devices.length > 1 && (
            <div className="select-wrap camera-select">
              <label htmlFor={liveSelectId} className="visually-hidden">Camera selector</label>
              <select
                id={liveSelectId}
                value={selectedDeviceId}
                onChange={(e) => {
                  setSelectedDeviceId(e.target.value);
                  if (isStreaming) {
                    stopCamera();
                    setTimeout(startCamera, 300);
                  }
                }}
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
              <ChevronDown size={14} />
            </div>
          )}

          <button
            type="button"
            className={`btn-ghost btn-sm ${mirrored ? "btn-active" : ""}`}
            onClick={() => setMirrored(!mirrored)}
            title="Toggle camera mirror flip"
          >
            <RefreshCw size={13} /> {mirrored ? "Mirrored" : "Normal"}
          </button>

          <button
            type="button"
            className={`btn-ghost btn-sm ${audioAlert ? "btn-active" : ""}`}
            onClick={() => setAudioAlert(!audioAlert)}
            title="Audio beep on high risk"
          >
            {audioAlert ? "🔊 Audio On" : "🔈 Audio Off"}
          </button>

          <button
            type="button"
            className="btn-ghost btn-sm"
            onClick={resetSession}
            disabled={!isStreaming || bufferCount === 0}
            title="Reset 32-frame sequence buffer"
          >
            <RefreshCw size={13} /> Reset 32F Buffer
          </button>

          {!isStreaming ? (
            <button
              type="button"
              className="btn-primary live-toggle-btn start"
              onClick={startCamera}
            >
              <Play size={15} fill="currentColor" /> Start Live Detection
            </button>
          ) : (
            <button
              type="button"
              className="btn-primary live-toggle-btn stop"
              onClick={stopCamera}
            >
              <Square size={14} fill="currentColor" /> Stop Webcam
            </button>
          )}
        </div>
      </div>

      {cameraError && (
        <div className="live-camera-error">
          <ShieldAlert size={18} />
          <div>
            <strong>Camera Access Notice:</strong>
            <p>{cameraError}</p>
          </div>
        </div>
      )}

      {/* Main Two-Column Console Grid */}
      <div className="live-grid">
        {/* Left: Video Feed with Face Tracking Canvas */}
        <div className="panel live-video-panel">
          <div className="live-video-container">
            <video
              ref={videoRef}
              className={`live-video-feed ${mirrored ? "mirrored" : ""}`}
              playsInline
              muted
              autoPlay
            />

            <canvas
              ref={overlayCanvasRef}
              className="live-overlay-canvas"
            />

            {!isStreaming && (
              <div className="live-video-placeholder">
                <div className="placeholder-icon">
                  <Camera size={44} strokeWidth={1.5} />
                </div>
                <h3>Live Face Deepfake Screening</h3>
                <p>
                  Click <strong>Start Live Detection</strong> to stream from your
                  laptop webcam. Real-time InsightFace SCRFD will detect your face
                  and feed a 32-frame sliding sequence to the MobileNetV3 + XGBoost
                  ensemble deepfake model.
                </p>
                <button
                  type="button"
                  className="btn-primary btn-lg"
                  onClick={startCamera}
                >
                  <Play size={16} fill="currentColor" /> Start Laptop Webcam
                </button>
                <div className="placeholder-specs">
                  <span><Check size={12} /> InsightFace SCRFD Detection</span>
                  <span><Check size={12} /> 32-Frame Temporal Sequence</span>
                  <span><Check size={12} /> Real MobileNet+XGBoost Inference</span>
                </div>
              </div>
            )}

            {isStreaming && (
              <div className="live-hud-top">
                <div className="hud-badge face-status">
                  <span
                    className={`status-circle ${
                      latestResult?.face_detected ? "detected" : "scanning"
                    }`}
                  />
                  {latestResult?.face_detected
                    ? `FACE DETECTED (${Math.round((latestResult.box?.confidence ?? 1) * 100)}%)`
                    : "SCANNING FOR FACE…"}
                </div>

                <div className="hud-badge detector-tag">
                  <span>{latestResult?.detector_name || "SCRFD"}</span>
                </div>
              </div>
            )}
          </div>

          {/* Under-video sequence thumb strip */}
          <div className="live-face-strip-bar">
            <div className="face-strip-header">
              <span>
                <Zap size={13} />
                <b>RECENT SEQUENCE CROPS</b> ({latestResult?.recent_crops?.length || 0})
              </span>
              <small>Normalized 224×224 patches fed to MobileNetV3</small>
            </div>

            <div className="face-strip-row">
              {latestResult?.recent_crops && latestResult.recent_crops.length > 0 ? (
                latestResult.recent_crops.map((thumb, idx) => (
                  <div key={idx} className="face-thumb-wrap">
                    <img src={thumb} alt={`Sequence crop ${idx + 1}`} />
                    <span className="thumb-idx">#{idx + 1}</span>
                  </div>
                ))
              ) : (
                <div className="strip-empty">
                  <span>Webcam face patches will appear here in real time</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Real-Time Risk Gauge, Buffer Monitor & Forensic Telemetry */}
        <div className="panel live-metrics-panel">
          <div className="panel-heading">
            <div>
              <div className="step-label">02 <span>·</span> CONTINUOUS INFERENCE</div>
              <h2>Deepfake Risk Telemetry</h2>
            </div>
            <div className="heading-icon">
              <FlaskConical size={18} />
            </div>
          </div>

          {/* Real-time Risk Score Card */}
          <div
            className={`live-risk-card ${
              isHighRisk ? "risk-high" : isSuspicious ? "risk-sus" : "risk-low"
            }`}
          >
            <div className="risk-card-left">
              <div className="risk-icon-wrap">
                {isHighRisk ? (
                  <ShieldAlert size={28} />
                ) : isSuspicious ? (
                  <AlertTriangle size={28} />
                ) : (
                  <Shield size={28} />
                )}
              </div>
              <div className="risk-copy">
                <span className="risk-eyebrow">
                  {latestResult?.risk_label || "AUTHENTIC / LOW RISK"}
                </span>
                <strong className="risk-verdict">
                  {isHighRisk
                    ? "Deepfake Signal Detected"
                    : isSuspicious
                    ? "Suspicious / Review Recommended"
                    : "Authentic Facial Motion"}
                </strong>
                <small className="risk-desc">
                  {isHighRisk
                    ? "MobileNetV3 + XGBoost sequence model detected high manipulation signals."
                    : isSuspicious
                    ? "Elevated manipulation probability. Maintain steady lighting and frontal angle."
                    : "Temporal facial dynamics and boundary consistency are within natural parameters."}
                </small>
              </div>
            </div>

            {/* Circular Risk Ring */}
            <div
              className={`live-score-gauge ${
                isHighRisk ? "gauge-high" : isSuspicious ? "gauge-sus" : "gauge-low"
              }`}
              style={
                {
                  "--gauge-score": `${Math.round(fakeScore * 100)}%`,
                } as React.CSSProperties
              }
            >
              <div className="gauge-inner">
                <span className="gauge-val">{Math.round(fakeScore * 100)}</span>
                <span className="gauge-pct">%</span>
                <small className="gauge-label">FAKE RISK</small>
              </div>
            </div>
          </div>

          {/* Threshold adjustment */}
          <div className="threshold-setting live-threshold">
            <div>
              <span>Flag Threshold</span>
              <b>{Math.round(threshold * 100)}%</b>
            </div>
            <input
              type="range"
              min="0.3"
              max="0.8"
              step="0.01"
              value={threshold}
              onChange={(e) => onThresholdChange(Number(e.target.value))}
            />
          </div>

          {/* 32-Frame Sequence Temporal Buffer Monitor */}
          <div className="live-buffer-monitor">
            <div className="buffer-header">
              <div>
                <strong>32-Frame Temporal Sequence</strong>
                <span className="buffer-state-label">
                  {bufferCount >= bufferTarget
                    ? "Active Sliding Window (Full)"
                    : `Buffering Warmup (${bufferCount}/${bufferTarget})`}
                </span>
              </div>
              <span className="buffer-counter">
                {bufferCount} / {bufferTarget}
              </span>
            </div>

            {/* 32 Segmented Progress Blocks */}
            <div className="buffer-segments">
              {Array.from({ length: bufferTarget }).map((_, i) => {
                const isFilled = i < bufferCount;
                const frameProb = latestResult?.frame_scores?.[i] ?? 0;
                let segClass = "empty";
                if (isFilled) {
                  segClass =
                    frameProb >= threshold
                      ? "high"
                      : frameProb >= 0.4
                      ? "sus"
                      : "low";
                }
                return (
                  <span
                    key={i}
                    className={`buffer-segment ${segClass}`}
                    title={`Frame ${i + 1}: ${
                      isFilled ? `${Math.round(frameProb * 100)}% fake risk` : "Pending"
                    }`}
                  />
                );
              })}
            </div>

            <div className="buffer-footer">
              <small>
                {bufferCount >= bufferTarget
                  ? "✓ Complete 32-frame sequence fed to MobileNet-XGBoost every frame"
                  : `Waiting for ${bufferTarget - bufferCount} more frames to achieve complete sliding temporal window`}
              </small>
            </div>
          </div>

          {/* Real-Time Score History Sparkline Chart */}
          <div className="live-history-box">
            <div className="history-header">
              <strong>Continuous Risk Score Timeline</strong>
              <small>{scoreHistory.length} frames logged</small>
            </div>

            <div className="history-chart">
              {scoreHistory.length > 1 ? (
                <svg viewBox="0 0 100 35" preserveAspectRatio="none">
                  {/* Threshold guideline */}
                  <line
                    x1="0"
                    x2="100"
                    y1={35 - threshold * 35}
                    y2={35 - threshold * 35}
                    className="timeline-threshold-line"
                  />
                  {/* Score polyline */}
                  <polyline
                    points={scoreHistory
                      .map(
                        (s, i) =>
                          `${(i / (scoreHistory.length - 1)) * 100},${35 - s * 35}`
                      )
                      .join(" ")}
                    className="timeline-score-line"
                  />
                </svg>
              ) : (
                <div className="history-empty">
                  <span>Timeline records continuous risk variations over time</span>
                </div>
              )}
            </div>
          </div>

          {/* Real-time Forensic Telemetry Grid */}
          <div className="live-telemetry-grid">
            <div className="telemetry-card">
              <span>FACE DETECTOR</span>
              <strong>{latestResult?.detector_name || "InsightFace SCRFD"}</strong>
              <small>{latestResult?.detect_ms ? `${latestResult.detect_ms} ms` : "0 ms"}</small>
            </div>

            <div className="telemetry-card">
              <span>SEQUENCE MODEL</span>
              <strong>MobileNetV3 + XGBoost</strong>
              <small>{latestResult?.inference_ms ? `${latestResult.inference_ms} ms` : "0 ms"}</small>
            </div>

            <div className="telemetry-card">
              <span>PROCESSED / FACES</span>
              <strong>{frameCount} frames</strong>
              <small>{faceCount} face crops</small>
            </div>

            <div className="telemetry-card">
              <span>WEBSOCKET LINK</span>
              <strong className={wsConnected ? "text-green" : "text-amber"}>
                {wsConnected ? "Connected" : "Idle"}
              </strong>
              <small>/api/ws/live</small>
            </div>
          </div>

          <div className="live-caveat-box">
            <CircleHelp size={14} />
            <span>
              Real-time deepfake analysis provides live screening cues for review.
              Hardware lighting, extreme angles, and low resolution can influence
              temporal scores.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
