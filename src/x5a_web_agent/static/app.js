(() => {
  const $ = (id) => document.getElementById(id);
  const talkBtn = $("talkBtn");
  const talkLabel = $("talkLabel");
  const transcriptEl = $("transcript");
  const parsedEl = $("parsed");
  const stageText = $("stageText");
  const stageCode = $("stageCode");
  const resultText = $("resultText");
  const linkBadge = $("linkBadge");
  const busyBadge = $("busyBadge");
  const holdingBadge = $("holdingBadge");
  const speechWarn = $("speechWarn");
  const speechStatus = $("speechStatus");
  const talkHint = $("talkHint");
  const textForm = $("textForm");
  const textInput = $("textInput");
  const stopBtn = $("stopBtn");
  const nativeRec = $("nativeRec");

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${location.host}/ws`;
  let ws = null;
  let reconnectTimer = null;
  let pollTimer = null;
  let holding = false;
  let wsFailed = false;
  let holdStarted = 0;
  let mediaStream = null;
  let mediaRecorder = null;
  let audioChunks = [];

  function setLink(ok, label) {
    linkBadge.textContent = label;
    linkBadge.className = `badge ${ok ? "ok" : "bad"}`;
  }

  function handleMessage(msg) {
    if (!msg) return;
    if (msg.type === "state") applyState(msg);
    if (msg.type === "transcript") transcriptEl.textContent = msg.text || "（空）";
    if (msg.type === "parsed" || msg.parsed) {
      parsedEl.textContent = JSON.stringify(msg.result || msg.parsed, null, 2);
    }
    if (msg.type === "status") {
      stageText.textContent = msg.stage_zh || msg.stage;
      stageCode.textContent = msg.stage || "";
      const terminal = ["SUCCESS", "FAILED", "IDLE", "DRY_RUN", "READY"].indexOf(msg.stage) >= 0;
      const busy = msg.busy !== false && !terminal;
      busyBadge.textContent = busy ? "忙碌" : "空闲";
      busyBadge.className = `pill ${busy ? "busy" : ""}`;
    }
    if (msg.type === "result") {
      resultText.textContent = msg.message || (msg.success ? "任务完成" : "任务失败");
      resultText.className = `result ${msg.success ? "" : "fail"}`;
      stageText.textContent = msg.stage_zh || msg.stage;
      stageCode.textContent = msg.stage || "";
      busyBadge.textContent = "空闲";
      busyBadge.className = "pill";
    }
    if (msg.type === "error" || msg.type === "info") {
      resultText.textContent = msg.message || "";
      resultText.className = `result ${msg.type === "error" ? "fail" : ""}`;
    }
  }

  function startPoll() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/api/state");
        if (!res.ok) return;
        applyState(await res.json());
        if (wsFailed || !ws || ws.readyState !== WebSocket.OPEN) {
          setLink(true, "已连接");
        }
      } catch (_err) {
        if (wsFailed) setLink(false, "已断开");
      }
    }, 700);
  }

  function connect() {
    startPoll();
    try {
      ws = new WebSocket(wsUrl);
    } catch (_err) {
      wsFailed = true;
      setLink(true, "已连接");
      return;
    }
    ws.onopen = () => {
      wsFailed = false;
      setLink(true, "已连接");
    };
    ws.onclose = () => {
      if (!wsFailed) setLink(false, "已断开");
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1500);
    };
    ws.onerror = () => {
      wsFailed = true;
      setLink(true, "已连接");
    };
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  }

  function applyHolding(holding, heldObject) {
    if (!holdingBadge) return;
    if (holding) {
      const label = (heldObject && heldObject.label) || "物体";
      holdingBadge.textContent = `拿着${label}`;
      holdingBadge.className = "pill busy";
    } else {
      holdingBadge.textContent = "空手";
      holdingBadge.className = "pill";
    }
  }

  function applyState(state) {
    if (state.transcript && !holding) transcriptEl.textContent = state.transcript;
    if (state.parsed) parsedEl.textContent = JSON.stringify(state.parsed, null, 2);
    if (state.stage) stageCode.textContent = state.stage;
    if (state.stage_zh) stageText.textContent = state.stage_zh;
    const observationSideBadge = $("observationSideBadge");
    if (observationSideBadge && state.observation_side) {
      observationSideBadge.textContent = state.observation_side === "mirror" ? "镜像侧" : "当前侧";
    }
    if (state.prepared !== undefined && holdingBadge) {
      const initialized = state.initialized === undefined ? state.prepared : state.initialized;
      holdingBadge.textContent = state.prepared ? "已到观察位" : (initialized ? "首次准备已完成" : "尚未准备");
      holdingBadge.className = `pill ${initialized ? "" : "busy"}`;
    }
    const terminal = ["SUCCESS", "FAILED", "IDLE", "DRY_RUN", "READY"].indexOf(state.stage) >= 0;
    const busy = !terminal && (Boolean(state.busy) || state.robot_state === "BUSY");
    busyBadge.textContent = busy ? "忙碌" : "空闲";
    busyBadge.className = `pill ${busy ? "busy" : ""}`;
    if (state.last_result && state.last_result.message) {
      resultText.textContent = state.last_result.message;
      resultText.className = `result ${state.last_result.success ? "" : "fail"}`;
    }
    if (state.server_ready === false) {
      setLink(false, "Task Server 未就绪");
    }
  }

  async function sendHttp(payload) {
    try {
      let res;
      if (payload.type === "text" || payload.type === "text_command") {
        res = await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: payload.text || "" }),
        });
      } else if (payload.type === "task" || payload.type === "manual_task") {
        res = await fetch("/api/task", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: payload.action || "pick_place",
            target_color: payload.target_color || "",
          }),
        });
      } else if (payload.type === "cancel") {
        res = await fetch("/api/cancel", { method: "POST" });
      } else {
        return;
      }
      handleMessage(await res.json());
      setLink(true, "已连接");
    } catch (_err) {
      setLink(false, "已断开");
      resultText.textContent = "网页未连接到主机";
      resultText.className = "result fail";
    }
  }

  function send(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      return;
    }
    sendHttp(payload);
  }

  function sendText(text) {
    const value = (text || "").trim();
    if (!value) return;
    transcriptEl.textContent = value;
    send({ type: "text_command", text: value });
  }

  function setSpeechStatus(text, kind) {
    speechStatus.textContent = text;
    speechStatus.className = `speech-status ${kind || ""}`;
  }

  function resetTalkButton() {
    talkBtn.classList.remove("listening", "sending");
    talkLabel.textContent = "按住 说话";
    talkHint.textContent = "松手发送 · 像 QQ 语音条";
  }

  function canWebRecord() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  }

  function openSystemRecorder() {
    if (nativeRec) nativeRec.click();
  }

  function pickRecorderMime() {
    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/ogg;codecs=opus",
    ];
    if (!window.MediaRecorder) return "";
    for (let i = 0; i < types.length; i += 1) {
      if (MediaRecorder.isTypeSupported(types[i])) return types[i];
    }
    return "";
  }

  function stopStream() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
  }

  async function beginHold(event) {
    event.preventDefault();
    if (holding) return;
    if (!canWebRecord()) {
      setSpeechStatus("网页录音不可用，已打开系统录音", "hot");
      openSystemRecorder();
      return;
    }
    if (location.protocol !== "https:" && location.hostname !== "127.0.0.1") {
      speechWarn.classList.remove("hidden");
      speechWarn.textContent = "录音需要 HTTPS。请用 https://192.168.0.50:8000 打开。";
    }
    holding = true;
    holdStarted = Date.now();
    audioChunks = [];
    talkBtn.classList.add("listening");
    talkBtn.classList.remove("sending");
    talkLabel.textContent = "松开发送";
    talkHint.textContent = "正在录音，不会中途断开";
    transcriptEl.textContent = "正在录音…";
    setSpeechStatus("按住中，请说话", "hot");
    if (navigator.vibrate) navigator.vibrate(20);
    try {
      talkBtn.setPointerCapture(event.pointerId);
    } catch (_err) {
      // ignore
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      const mime = pickRecorderMime();
      mediaRecorder = mime
        ? new MediaRecorder(mediaStream, { mimeType: mime })
        : new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) audioChunks.push(ev.data);
      };
      mediaRecorder.start(200);
    } catch (_err) {
      holding = false;
      stopStream();
      resetTalkButton();
      setSpeechStatus("麦克风被拒绝，请允许权限后重试", "hot");
    }
  }

  async function uploadAndSend(blob) {
    talkBtn.classList.add("sending");
    talkLabel.textContent = "正在识别";
    talkHint.textContent = "主机 faster-whisper";
    setSpeechStatus("正在用本地模型识别…", "hot");
    transcriptEl.textContent = "正在识别…";
    const body = new FormData();
    const ext = (blob.type || "").includes("mp4") ? "m4a" : "webm";
    body.append("audio", blob, `speech.${ext}`);
    const res = await fetch("/api/asr", { method: "POST", body });
    const data = await res.json();
    const text = (data.text || "").trim();
    if (!res.ok || !text) {
      resetTalkButton();
      setSpeechStatus(data.message || "没有听清，请再说一次", "hot");
      transcriptEl.textContent = data.message || "没有听清";
      return;
    }
    talkLabel.textContent = "已发送";
    talkHint.textContent = text;
    setSpeechStatus(`识别到：${text}`, "send");
    transcriptEl.textContent = text;
    if (navigator.vibrate) navigator.vibrate([15, 40, 15]);
    sendText(text);
    setTimeout(resetTalkButton, 900);
  }

  function endHold(event) {
    if (!holding) return;
    event.preventDefault();
    holding = false;
    const elapsed = Date.now() - holdStarted;
    talkBtn.classList.remove("listening");
    const recorder = mediaRecorder;
    mediaRecorder = null;
    if (elapsed < 500) {
      if (recorder && recorder.state !== "inactive") recorder.stop();
      stopStream();
      resetTalkButton();
      setSpeechStatus("按太短了，请按住说完整一句", "hot");
      transcriptEl.textContent = "按住说话，松手发送";
      return;
    }
    if (!recorder) {
      stopStream();
      resetTalkButton();
      setSpeechStatus("没有录到声音", "hot");
      return;
    }
    recorder.onstop = () => {
      const type = recorder.mimeType || "audio/webm";
      const blob = new Blob(audioChunks, { type });
      stopStream();
      if (blob.size < 800) {
        resetTalkButton();
        setSpeechStatus("录音太短，请再说一次", "hot");
        return;
      }
      uploadAndSend(blob).catch((err) => {
        resetTalkButton();
        setSpeechStatus(`识别失败：${err}`, "hot");
      });
    };
    if (recorder.state !== "inactive") recorder.stop();
    else recorder.onstop();
  }

  talkBtn.addEventListener("pointerdown", beginHold);
  talkBtn.addEventListener("pointerup", endHold);
  talkBtn.addEventListener("pointercancel", endHold);
  talkBtn.addEventListener("contextmenu", (event) => event.preventDefault());

  if (nativeRec) {
    nativeRec.addEventListener("change", () => {
      const file = nativeRec.files && nativeRec.files[0];
      nativeRec.value = "";
      if (!file) return;
      uploadAndSend(file).catch((err) => {
        resetTalkButton();
        setSpeechStatus(`识别失败：${err}`, "hot");
      });
    });
  }

  textForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendText(textInput.value);
    textInput.value = "";
  });

  document.querySelectorAll("[data-command]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const command = btn.getAttribute("data-command");
      transcriptEl.textContent = `手动：${command}`;
      sendText(command);
    });
  });

  stopBtn.addEventListener("click", () => send({ type: "cancel" }));

  const ua = navigator.userAgent || "";
  const inQQOrWeChat = /QQ\//.test(ua) || /MicroMessenger/.test(ua);
  if (inQQOrWeChat) {
    speechWarn.classList.remove("hidden");
    speechWarn.textContent =
      "QQ/微信内置浏览器经常不能网页录音。请点右上角用系统浏览器打开，或直接点「系统录音」。";
  } else if (location.protocol !== "https:") {
    speechWarn.classList.remove("hidden");
    speechWarn.textContent =
      "当前是 HTTP，手机一般不能录音。请改用 https://192.168.0.50:8000";
  } else if (!canWebRecord()) {
    speechWarn.classList.remove("hidden");
    speechWarn.textContent =
      "这个浏览器不能网页录音。请点「系统录音」，或用手机自带 Chrome 打开。";
  }

  connect();
})();
