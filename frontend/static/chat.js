const btn = document.getElementById('btn');

const _proto  = window.location.protocol;
const _wsProto = _proto === 'https:' ? 'wss:' : 'ws:';
const _host   = window.location.host;
const API     = `${_proto}//${_host}`;
const WS      = `${_wsProto}//${_host}`;

/* ── Audio config ─────────────────────────────────────────────────────────── */
const MIC_SAMPLE_RATE  = 16_000;   // sent to server (Deepgram Flux)
const BOT_SAMPLE_RATE  = 24_000;   // received from server (Cartesia sonic-2)

/* ── State ────────────────────────────────────────────────────────────────── */
let isRunning       = false;
let audioContext    = null;
let mediaStream     = null;
let audioWorklet    = null;
let audioWs         = null;
let playbackCtx     = null;
let nextPlayTime    = 0;
let conversation    = [];
let attempt         = 0;
const MAX_ATTEMPTS  = 3;
const RETRYABLE_WS_CODES = new Set([1006, 1008, 1011, 1012, 1013]);

/* ── UI helpers (these delegate to window.uiSet* defined in index.html) ──── */
function setIdle ()        { window.uiSetIdle?.();        }
function setActive ()      { window.uiSetActive?.();      }
function setBotSpeaking () { window.uiSetBotSpeaking?.(); }
function setConnecting ()  { window.uiSetConnecting?.();  }

/* ── Audio capture + WebSocket stream ───────────────────────────────────── */
async function startAudioCapture () {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: MIC_SAMPLE_RATE,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext({ sampleRate: MIC_SAMPLE_RATE });
  const actualRate = audioContext.sampleRate;
  console.log(`[audio] context: ${actualRate} Hz → target: ${MIC_SAMPLE_RATE} Hz`);

  const source = audioContext.createMediaStreamSource(mediaStream);

  await audioContext.audioWorklet.addModule(`/static/pcm-processor.js?v=${Date.now()}`);
  audioWorklet = new AudioWorkletNode(audioContext, 'audio-in-processor', {
    processorOptions: {
      inputSampleRate:  actualRate,
      outputSampleRate: MIC_SAMPLE_RATE,
    },
  });

  /* ── Outgoing: mic PCM → server ── */
  audioWs = new WebSocket(`${WS}/audio`);
  audioWs.binaryType = 'arraybuffer';

  audioWs.onopen = () => {
    attempt = 0;
    console.log('[audio] WS open — streaming Int16 PCM @', MIC_SAMPLE_RATE, 'Hz');
  };

  audioWs.onerror = (e) => console.error('[audio] WS error:', e);

  audioWs.onclose = async (e) => {
    console.log('[audio] WS closed:', e.code, e.reason);
    if (RETRYABLE_WS_CODES.has(e.code) && isRunning && attempt < MAX_ATTEMPTS) {
      attempt++;
      console.log(`[audio] Reconnect attempt ${attempt}/${MAX_ATTEMPTS}…`);
      await handleResume();
    }
  };

  /* ── Incoming: server PCM → speakers ── */
  playbackCtx  = new AudioContext({ sampleRate: BOT_SAMPLE_RATE });
  nextPlayTime = 0;

  audioWs.onmessage = (e) => {
    if (!(e.data instanceof ArrayBuffer) || e.data.byteLength === 0) return;

    const int16   = new Int16Array(e.data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 0x8000;
    }

    const buffer = playbackCtx.createBuffer(1, float32.length, BOT_SAMPLE_RATE);
    buffer.getChannelData(0).set(float32);

    const src = playbackCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(playbackCtx.destination);

    const startAt = Math.max(playbackCtx.currentTime, nextPlayTime);
    src.start(startAt);
    nextPlayTime = startAt + buffer.duration;

    setBotSpeaking();
    src.onended = () => {
      if (nextPlayTime <= playbackCtx.currentTime) setActive();
    };
  };

  /* ── Wire worklet output → WebSocket ── */
  audioWorklet.port.onmessage = (e) => {
    if (audioWs && audioWs.readyState === WebSocket.OPEN) {
      audioWs.send(e.data);
    }
  };

  /* Silence node: keeps AudioContext alive without routing to speakers */
  const silentGain     = audioContext.createGain();
  silentGain.gain.value = 0;
  source.connect(audioWorklet);
  audioWorklet.connect(silentGain);
  silentGain.connect(audioContext.destination);
}

/* ── Reconnect with context ──────────────────────────────────────────────── */
async function handleResume () {
  const contextText = conversation
    .slice(-20)
    .map(m => `${m.role === 'bot' ? 'Assistant' : 'User'}: ${m.text}`)
    .join('\n');

  /* Tear down old audio infrastructure */
  audioWorklet?.disconnect(); audioWorklet = null;
  audioContext?.close();     audioContext  = null;
  playbackCtx?.close();      playbackCtx   = null;
  audioWs = null;

  await new Promise(r => setTimeout(r, 1500));

  setConnecting();
  await startAudioCapture();

  /* Ask server to greet again so pipeline is ready */
  await fetch(`${API}/greet`);
  await new Promise(r => setTimeout(r, 500));

  /* Inject previous conversation context */
  if (contextText) {
    await fetch(`${API}/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context: contextText }),
    });
    console.log('[resume] context injected');
  }
  setActive();
}

/* ── Stop everything ─────────────────────────────────────────────────────── */
function stopAudioCapture () {
  audioWorklet?.disconnect(); audioWorklet = null;
  audioContext?.close();     audioContext  = null;
  mediaStream?.getTracks().forEach(t => t.stop()); mediaStream = null;
  audioWs?.close();          audioWs       = null;
  playbackCtx?.close();      playbackCtx   = null;
  nextPlayTime = 0;
  console.log('[audio] capture stopped');
}

/* ── Button toggle ───────────────────────────────────────────────────────── */
btn.addEventListener('click', async () => {
  if (isRunning) {
    /* ── STOP ── */
    btn.disabled = true;
    stopAudioCapture();
    await fetch(`${API}/restart`).catch(() => {});
    window.uiClearTranscript?.();
    isRunning    = false;
    conversation = [];
    attempt      = 0;
    setIdle();
    btn.disabled = false;
    return;
  }

  /* ── START ── */
  setConnecting();
  isRunning = true;

  try {
    await startAudioCapture();
  } catch (err) {
    console.error('[audio] capture failed:', err);
    setIdle();
    isRunning = false;
    return;
  }

  /* Small delay to let the WebSocket fully open before greeting */
  await new Promise(r => setTimeout(r, 600));

  try {
    await fetch(`${API}/greet`);
    console.log('[greet] sent');
  } catch (e) {
    console.error('[greet] error:', e);
  }

  setActive();
});