/**
 * VoiceGuard AI — Frontend Application Logic
 * Lightweight Synthetic Voice & Deepfake Audio Detection Platform
 */

// Application State
const state = {
  currentView: 'overview',
  activeAudioFile: null,
  activeAudioBlob: null,
  activeModel: 'best_model',
  isRecording: false,
  isPaused: false,
  mediaRecorder: null,
  audioChunks: [],
  audioContext: null,
  analyserNode: null,
  animationFrameId: null,
  recordingTimerInterval: null,
  recordingSeconds: 0,
  historyPage: 0,
  historyLimit: 15,
  historyFilter: '',
  historySearch: '',
  trainingPollInterval: null,
  charts: {
    ratioChart: null,
    rocChart: null,
    latencyChart: null
  }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initAudioInput();
  initMicrophone();
  initHistoryControls();
  initTrainingControls();
  initSettings();

  // Load initial system data
  checkSystemHealth();
  loadDashboardStats();
  
  // Handle hash changes
  window.addEventListener('hashchange', handleHashChange);
  if (window.location.hash) {
    handleHashChange();
  }
});

// --- Toast Notifications ---
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.innerHTML = `<span>${type === 'error' ? '⚠' : '✓'}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// --- Navigation & View Switching ---
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      const view = item.getAttribute('data-view');
      if (view) {
        switchView(view);
      }
    });
  });

  const btnQuick = document.getElementById('btnQuickAnalyze');
  if (btnQuick) {
    btnQuick.addEventListener('click', () => switchView('analyzer'));
  }
}

function handleHashChange() {
  const hash = window.location.hash.replace('#', '');
  if (hash && ['overview', 'analyzer', 'history', 'analytics', 'models', 'dataset', 'settings'].includes(hash)) {
    switchView(hash);
  }
}

function switchView(viewName) {
  state.currentView = viewName;

  // Update Sidebar active state
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.getAttribute('data-view') === viewName);
  });

  // Update View containers
  document.querySelectorAll('.view-content').forEach(view => {
    view.classList.toggle('active-view', view.id === `view-${viewName}`);
  });

  // Update Breadcrumbs & Header Title
  const titles = {
    overview: { title: 'Operational Dashboard', breadcrumb: 'Forensic Suite / Overview' },
    analyzer: { title: 'Voice Analyzer', breadcrumb: 'Forensic Suite / Acoustic Analysis' },
    history: { title: 'Detection History', breadcrumb: 'Forensic Suite / Audit Log' },
    analytics: { title: 'Research Analytics', breadcrumb: 'Forensic Suite / Benchmark Engine' },
    models: { title: 'Model Management', breadcrumb: 'Engine & Data / Classifiers' },
    dataset: { title: 'ASVspoof 2019 LA Dataset', breadcrumb: 'Engine & Data / Corpus' },
    settings: { title: 'System Settings', breadcrumb: 'Configuration / Preferences' }
  };

  const info = titles[viewName] || titles.overview;
  document.getElementById('viewHeaderTitle').textContent = info.title;
  document.getElementById('viewBreadcrumb').textContent = info.breadcrumb;
  window.location.hash = viewName;

  // Lazy load data for the switched view
  if (viewName === 'overview') loadDashboardStats();
  else if (viewName === 'history') loadHistory();
  else if (viewName === 'analytics') loadResearchAnalytics();
  else if (viewName === 'models') loadModelsView();
  else if (viewName === 'dataset') loadDatasetInfo();
}

// --- System Health Check ---
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = document.getElementById('headerModelStatusDot');
    const nameEl = document.getElementById('headerModelName');
    const sideStatus = document.getElementById('sidebarStatusText');

    if (data.model_ready) {
      dot.className = 'status-indicator';
      nameEl.textContent = data.active_model || 'Model Ready';
      if (sideStatus) sideStatus.textContent = 'Engine Active';
    } else {
      dot.className = 'status-indicator offline';
      nameEl.textContent = 'Uninitialized (Train Model)';
      if (sideStatus) sideStatus.textContent = 'Model Setup Required';
    }
  } catch (err) {
    console.warn('Health check error:', err);
  }
}

// --- Module A: Overview & Stats ---
async function loadDashboardStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    const data = await res.json();

    document.getElementById('statTotal').textContent = data.total.toLocaleString();
    document.getElementById('statSynthetic').textContent = data.synthetic.toLocaleString();
    document.getElementById('statGenuine').textContent = data.genuine.toLocaleString();
    document.getElementById('statSyntheticPct').textContent = `${data.synthetic_pct}% of analyses`;
    document.getElementById('statGenuinePct').textContent = `${data.genuine_pct}% of analyses`;
    document.getElementById('statLatency').textContent = `${data.avg_latency_ms.toFixed(1)} ms`;

    // Render Doughnut Chart
    renderOverviewRatioChart(data.genuine, data.synthetic);

    // Render Recent Table
    renderOverviewRecentTable(data.recent || []);
  } catch (err) {
    console.error('Error loading dashboard stats:', err);
  }
}

function renderOverviewRatioChart(genuine, synthetic) {
  const canvas = document.getElementById('overviewRatioChart');
  if (!canvas) return;

  if (state.charts.ratioChart) {
    state.charts.ratioChart.destroy();
  }

  const total = genuine + synthetic;
  const dataValues = total === 0 ? [1] : [genuine, synthetic];
  const bgColors = total === 0 ? ['#24333C'] : ['#C6FF00', '#FF6475'];

  state.charts.ratioChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: total === 0 ? ['No Data'] : ['Genuine', 'Synthetic'],
      datasets: [{
        data: dataValues,
        backgroundColor: bgColors,
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '76%',
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: total > 0,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} (${((ctx.raw / total) * 100).toFixed(1)}%)`
          }
        }
      }
    }
  });
}

function renderOverviewRecentTable(items) {
  const tbody = document.getElementById('overviewRecentTable');
  if (!tbody) return;

  if (items.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No voice analyses recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = items.map(item => {
    const isSynthetic = item.label === 'synthetic';
    const badgeClass = isSynthetic ? 'badge-synthetic' : 'badge-genuine';
    const dateFormatted = new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `
      <tr>
        <td style="font-family: var(--font-mono); font-size: 12px;">${escapeHtml(item.filename)}</td>
        <td><span class="badge ${badgeClass}">${item.label.toUpperCase()}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 700;">${(item.confidence * 100).toFixed(1)}%</td>
        <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${dateFormatted}</td>
      </tr>
    `;
  }).join('');
}

// --- Module B & C: Audio Upload & Microphone Recording ---
function initAudioInput() {
  const dropzone = document.getElementById('audioDropzone');
  const fileInput = document.getElementById('audioFileInput');
  const btnRemove = document.getElementById('btnRemoveAudio');
  const btnAnalyze = document.getElementById('btnRunAnalysis');

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedAudioFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleSelectedAudioFile(fileInput.files[0]);
    }
  });

  btnRemove.addEventListener('click', clearActiveAudio);
  btnAnalyze.addEventListener('click', executeAnalysis);
}

function handleSelectedAudioFile(file) {
  state.activeAudioFile = file;
  state.activeAudioBlob = null;

  const audioEl = document.getElementById('audioElement');
  const previewBox = document.getElementById('audioPreviewBox');
  const fileNameEl = document.getElementById('previewFileName');
  const fileMetaEl = document.getElementById('previewFileMeta');
  const btnAnalyze = document.getElementById('btnRunAnalysis');

  const fileUrl = URL.createObjectURL(file);
  audioEl.src = fileUrl;
  fileNameEl.textContent = file.name;
  
  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  fileMetaEl.textContent = `Size: ${sizeMb} MB | Format: ${file.name.split('.').pop().toUpperCase()}`;

  previewBox.style.display = 'block';
  btnAnalyze.disabled = false;
  showToast(`Loaded audio file: ${file.name}`, 'info');
}

function clearActiveAudio() {
  state.activeAudioFile = null;
  state.activeAudioBlob = null;

  const audioEl = document.getElementById('audioElement');
  audioEl.pause();
  audioEl.src = '';

  document.getElementById('audioPreviewBox').style.display = 'none';
  document.getElementById('btnRunAnalysis').disabled = true;
  document.getElementById('audioFileInput').value = '';
}

// Function to encode Float32 PCM samples into standard 16-bit PCM WAV Blob
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

// Microphone Recording using Web Audio API Float32 PCM & standard WAV encoding
function initMicrophone() {
  const btnStart = document.getElementById('btnRecordStart');
  const btnPause = document.getElementById('btnRecordPause');
  const btnStop = document.getElementById('btnRecordStop');
  const timerEl = document.getElementById('micTimer');

  btnStart.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      startRecording(stream);
    } catch (err) {
      showToast(`Microphone permission denied: ${err.message}`, 'error');
    }
  });

  btnPause.addEventListener('click', () => {
    if (!state.isRecording) return;
    state.isPaused = !state.isPaused;
    btnPause.textContent = state.isPaused ? 'Resume' : 'Pause';
    if (state.isPaused) {
      timerEl.classList.remove('recording');
    } else {
      timerEl.classList.add('recording');
    }
  });

  btnStop.addEventListener('click', () => {
    if (state.isRecording) {
      stopMicrophoneRecording();
    }
  });
}

function startRecording(stream) {
  state.isRecording = true;
  state.isPaused = false;
  state.pcmBuffers = [];
  state.recordingSeconds = 0;
  state.activeStream = stream;

  // Set up Web Audio Analyser for live visualizer and raw PCM sample capture
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new AudioContextClass();
  const source = state.audioContext.createMediaStreamSource(stream);

  state.analyserNode = state.audioContext.createAnalyser();
  state.analyserNode.fftSize = 256;
  source.connect(state.analyserNode);

  drawMicWaveform();

  // Buffer raw PCM audio samples for WAV encoding
  const bufferSize = 4096;
  state.pcmProcessor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
  state.pcmProcessor.onaudioprocess = (e) => {
    if (state.isRecording && !state.isPaused) {
      const channel = e.inputBuffer.getChannelData(0);
      state.pcmBuffers.push(new Float32Array(channel));
    }
  };
  source.connect(state.pcmProcessor);
  state.pcmProcessor.connect(state.audioContext.destination);

  // Update UI buttons & timer
  document.getElementById('btnRecordStart').style.display = 'none';
  document.getElementById('btnRecordPause').style.display = 'inline-flex';
  document.getElementById('btnRecordPause').textContent = 'Pause';
  document.getElementById('btnRecordStop').style.display = 'inline-flex';

  const timerEl = document.getElementById('micTimer');
  timerEl.classList.add('recording');

  state.recordingTimerInterval = setInterval(() => {
    if (!state.isPaused) {
      state.recordingSeconds++;
      const mins = String(Math.floor(state.recordingSeconds / 60)).padStart(2, '0');
      const secs = String(state.recordingSeconds % 60).padStart(2, '0');
      timerEl.textContent = `${mins}:${secs}`;
      if (state.recordingSeconds >= 30) {
        stopMicrophoneRecording();
      }
    }
  }, 1000);
}

function stopMicrophoneRecording() {
  state.isRecording = false;
  state.isPaused = false;

  if (state.activeStream) {
    state.activeStream.getTracks().forEach(track => track.stop());
    state.activeStream = null;
  }
  if (state.pcmProcessor) {
    state.pcmProcessor.disconnect();
    state.pcmProcessor = null;
  }
  if (state.analyserNode) {
    state.analyserNode.disconnect();
  }

  cancelAnimationFrame(state.animationFrameId);
  clearInterval(state.recordingTimerInterval);

  // Reset UI
  document.getElementById('btnRecordStart').style.display = 'inline-flex';
  document.getElementById('btnRecordPause').style.display = 'none';
  document.getElementById('btnRecordStop').style.display = 'none';
  const timerEl = document.getElementById('micTimer');
  timerEl.classList.remove('recording');

  let totalSamples = 0;
  for (const b of state.pcmBuffers) totalSamples += b.length;

  if (totalSamples === 0) {
    showToast('No audio captured. Please speak into the microphone.', 'error');
    if (state.audioContext) {
      state.audioContext.close();
      state.audioContext = null;
    }
    return;
  }

  const merged = new Float32Array(totalSamples);
  let offset = 0;
  for (const b of state.pcmBuffers) {
    merged.set(b, offset);
    offset += b.length;
  }

  const sampleRate = state.audioContext ? state.audioContext.sampleRate : 16000;
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }

  const wavBlob = encodeWAV(merged, sampleRate);
  const recordedFile = new File([wavBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });

  handleSelectedAudioFile(recordedFile);
  showToast('Voice recording captured successfully as PCM WAV!', 'info');
}

function drawMicWaveform() {
  const canvas = document.getElementById('micWaveformCanvas');
  if (!canvas || !state.analyserNode) return;
  const ctx = canvas.getContext('2d');
  const bufferLength = state.analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const draw = () => {
    state.animationFrameId = requestAnimationFrame(draw);
    state.analyserNode.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#080D12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#C6FF00';
    ctx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  };

  draw();
}

// --- Module D & E: Voice Detection Execution & Visualizations ---
async function executeAnalysis() {
  if (!state.activeAudioFile) {
    showToast('Please upload or record an audio sample first.', 'error');
    return;
  }

  const modelSelect = document.getElementById('analyzerModelSelect');
  const selectedModel = modelSelect ? modelSelect.value : 'best_model';

  const awaitingBox = document.getElementById('resultAwaiting');
  const loadingBox = document.getElementById('resultLoading');
  const containerBox = document.getElementById('resultContainer');
  const btnAnalyze = document.getElementById('btnRunAnalysis');

  awaitingBox.style.display = 'none';
  containerBox.style.display = 'none';
  loadingBox.style.display = 'block';
  btnAnalyze.disabled = true;

  const formData = new FormData();
  formData.append('audio', state.activeAudioFile);
  formData.append('model', selectedModel);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP error ${res.status}`);
    }

    renderAnalysisResult(data);
    showToast(`Analysis complete: ${data.label.toUpperCase()}`, data.is_synthetic ? 'error' : 'info');
  } catch (err) {
    loadingBox.style.display = 'none';
    awaitingBox.style.display = 'block';
    btnAnalyze.disabled = false;
    showToast(`Analysis failed: ${err.message}`, 'error');
  }
}

function renderAnalysisResult(data) {
  const loadingBox = document.getElementById('resultLoading');
  const containerBox = document.getElementById('resultContainer');
  const btnAnalyze = document.getElementById('btnRunAnalysis');

  loadingBox.style.display = 'none';
  containerBox.style.display = 'block';
  btnAnalyze.disabled = false;

  // Header Badge
  const badge = document.getElementById('resultBadge');
  const isSynthetic = data.is_synthetic;
  badge.className = `result-header-badge ${isSynthetic ? 'synthetic' : 'genuine'}`;
  badge.textContent = isSynthetic ? 'SYNTHETIC SPEECH DETECTED' : 'BONAFIDE HUMAN SPEECH';

  // Confidence & Scores
  document.getElementById('resultConfidenceVal').textContent = `${data.confidence_pct}%`;
  document.getElementById('scoreValGenuine').textContent = `${data.genuine_pct}%`;
  document.getElementById('scoreValSynthetic').textContent = `${data.synthetic_pct}%`;

  const barGen = document.getElementById('scoreBarGenuine');
  const barSyn = document.getElementById('scoreBarSynthetic');
  barGen.style.width = `${data.genuine_pct}%`;
  barSyn.style.width = `${data.synthetic_pct}%`;

  // Notice text
  const noticeBox = document.getElementById('resultNoticeText');
  noticeBox.className = `result-notice ${isSynthetic ? 'synthetic' : 'genuine'}`;
  noticeBox.textContent = data.explanation;

  // Metadata Grid
  document.getElementById('resultModelName').textContent = data.model_name;
  document.getElementById('resultLatency').textContent = `${data.latency_ms} ms`;
  document.getElementById('resultInferenceMs').textContent = `${data.inference_ms} ms`;
  document.getElementById('resultDuration').textContent = `${data.audio_duration} s`;
  document.getElementById('resultId').textContent = data.id;

  const timeStr = new Date(data.created_at).toLocaleTimeString();
  document.getElementById('resultTimestamp').textContent = `Analyzed at ${timeStr}`;

  // Render Visualizations if available
  if (data.visuals) {
    renderAudioVisualizations(data.visuals);
  }
}

function renderAudioVisualizations(vis) {
  const visSection = document.getElementById('visualizationSection');
  visSection.style.display = 'flex';

  document.getElementById('visDurationText').textContent = `${vis.duration} seconds (${vis.sampling_rate} Hz)`;

  // 1. Waveform Canvas
  const waveCanvas = document.getElementById('visWaveformCanvas');
  if (waveCanvas && vis.waveform) {
    drawWaveformCanvas(waveCanvas, vis.waveform);
  }

  // 2. Spectrogram Canvas
  const specCanvas = document.getElementById('visSpectrogramCanvas');
  if (specCanvas && vis.mel_spectrogram) {
    drawSpectrogramCanvas(specCanvas, vis.mel_spectrogram.matrix);
  }

  // 3. MFCC Heatmap Canvas
  const mfccCanvas = document.getElementById('visMfccCanvas');
  if (mfccCanvas && vis.mfcc_heatmap) {
    drawMfccCanvas(mfccCanvas, vis.mfcc_heatmap.matrix);
  }
}

function drawWaveformCanvas(canvas, peaks) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth;
  const h = canvas.height = canvas.offsetHeight;
  ctx.fillStyle = '#080D12';
  ctx.fillRect(0, 0, w, h);

  const barWidth = Math.max(2, (w / peaks.length) - 1);
  ctx.fillStyle = '#00E5FF';

  for (let i = 0; i < peaks.length; i++) {
    const x = i * (w / peaks.length);
    const amp = Math.min(1.0, peaks[i]);
    const barHeight = Math.max(3, amp * (h - 20));
    const y = (h - barHeight) / 2;
    ctx.fillRect(x, y, barWidth, barHeight);
  }
}

function drawSpectrogramCanvas(canvas, matrix) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth;
  const h = canvas.height = canvas.offsetHeight;
  ctx.fillStyle = '#080D12';
  ctx.fillRect(0, 0, w, h);

  const rows = matrix.length; // 40 mels
  if (rows === 0) return;
  const cols = matrix[0].length;
  const cellW = w / cols;
  const cellH = h / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = matrix[r][c]; // in dB, typically -80 to 0
      const norm = Math.max(0, Math.min(1, (val + 80) / 80));
      // Heat color palette (purple -> cyan -> lime)
      const red = Math.floor(norm * 198);
      const green = Math.floor(norm * 255);
      const blue = Math.floor((1 - norm) * 200);
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(c * cellW, h - (r + 1) * cellH, cellW + 1, cellH + 1);
    }
  }
}

function drawMfccCanvas(canvas, matrix) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth;
  const h = canvas.height = canvas.offsetHeight;
  ctx.fillStyle = '#080D12';
  ctx.fillRect(0, 0, w, h);

  const rows = matrix.length; // 20 mfccs
  if (rows === 0) return;
  const cols = matrix[0].length;
  const cellW = w / cols;
  const cellH = h / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = matrix[r][c];
      const norm = Math.max(0, Math.min(1, (val + 100) / 200));
      ctx.fillStyle = `rgba(198, 255, 0, ${norm})`;
      ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
    }
  }
}

// --- Module F: Detection History ---
function initHistoryControls() {
  const searchInput = document.getElementById('historySearchInput');
  const filterBtns = document.querySelectorAll('.filter-btn');
  const btnPrev = document.getElementById('btnHistPrev');
  const btnNext = document.getElementById('btnHistNext');
  const btnClear = document.getElementById('btnClearHistory');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.historyFilter = btn.getAttribute('data-filter') || '';
      state.historyPage = 0;
      loadHistory();
    });
  });

  let searchTimeout = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.historySearch = searchInput.value.trim();
      state.historyPage = 0;
      loadHistory();
    }, 300);
  });

  btnPrev.addEventListener('click', () => {
    if (state.historyPage > 0) {
      state.historyPage--;
      loadHistory();
    }
  });

  btnNext.addEventListener('click', () => {
    state.historyPage++;
    loadHistory();
  });

  btnClear.addEventListener('click', async () => {
    if (confirm('Are you sure you want to permanently clear all detection audit logs?')) {
      try {
        const res = await fetch('/api/history', { method: 'DELETE' });
        const data = await res.json();
        showToast(`Cleared ${data.deleted_count} history records.`, 'info');
        loadHistory();
      } catch (err) {
        showToast(`Failed to clear history: ${err.message}`, 'error');
      }
    }
  });
}

async function loadHistory() {
  const tbody = document.getElementById('historyTableBody');
  const countText = document.getElementById('historyCountText');
  const offset = state.historyPage * state.historyLimit;

  try {
    const params = new URLSearchParams({
      limit: state.historyLimit,
      offset: offset
    });
    if (state.historyFilter) params.append('label', state.historyFilter);
    if (state.historySearch) params.append('q', state.historySearch);

    const res = await fetch(`/api/history?${params.toString()}`);
    const data = await res.json();

    const items = data.items || [];
    const total = data.total || 0;

    countText.textContent = `Showing ${items.length > 0 ? offset + 1 : 0} to ${Math.min(offset + items.length, total)} of ${total} records`;
    document.getElementById('btnHistPrev').disabled = state.historyPage === 0;
    document.getElementById('btnHistNext').disabled = offset + items.length >= total;

    if (items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 24px;">No records match your filters.</td></tr>';
      return;
    }

    tbody.innerHTML = items.map(item => {
      const isSynthetic = item.label === 'synthetic';
      const badgeClass = isSynthetic ? 'badge-synthetic' : 'badge-genuine';
      const createdStr = new Date(item.created_at).toLocaleString();
      return `
        <tr>
          <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${item.id}</td>
          <td style="font-family: var(--font-mono); font-size: 11px;">${createdStr}</td>
          <td style="font-family: var(--font-mono); font-size: 12px; font-weight: 600;">${escapeHtml(item.filename)}</td>
          <td><span class="badge ${badgeClass}">${item.label.toUpperCase()}</span></td>
          <td style="font-family: var(--font-mono); font-weight: 700;">${(item.confidence * 100).toFixed(1)}%</td>
          <td style="font-family: var(--font-mono); color: var(--danger-red);">${(item.synthetic_score * 100).toFixed(1)}%</td>
          <td style="font-size: 11px; color: var(--text-secondary);">${escapeHtml(item.model_name)}</td>
          <td style="font-family: var(--font-mono); font-size: 11px;">${item.latency_ms.toFixed(1)}ms</td>
          <td>
            <button class="btn-secondary" style="padding: 2px 8px; font-size: 10px;" onclick="inspectHistoryItem('${item.id}')">Inspect</button>
            <button class="btn-danger" style="padding: 2px 6px; font-size: 10px; margin-left: 4px;" onclick="deleteHistoryItem('${item.id}')">✕</button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--danger-red);">Failed to load history: ${err.message}</td></tr>`;
  }
}

async function inspectHistoryItem(id) {
  try {
    const res = await fetch(`/api/history/${id}`);
    if (!res.ok) throw new Error('Analysis not found');
    const item = await res.json();

    const modalBody = document.getElementById('modalBody');
    modalBody.innerHTML = `
      <div><span style="color: var(--text-muted);">RECORD ID:</span> <b style="color: var(--accent-lime);">${item.id}</b></div>
      <div><span style="color: var(--text-muted);">FILENAME:</span> ${escapeHtml(item.filename)}</div>
      <div><span style="color: var(--text-muted);">TIMESTAMP:</span> ${new Date(item.created_at).toUTCString()}</div>
      <div><span style="color: var(--text-muted);">CLASSIFICATION:</span> <span class="badge ${item.label === 'synthetic' ? 'badge-synthetic' : 'badge-genuine'}">${item.label.toUpperCase()}</span></div>
      <div><span style="color: var(--text-muted);">MODEL CONFIDENCE:</span> ${(item.confidence * 100).toFixed(2)}%</div>
      <div><span style="color: var(--text-muted);">SPOOF SCORE:</span> ${(item.synthetic_score * 100).toFixed(2)}%</div>
      <div><span style="color: var(--text-muted);">BONAFIDE SCORE:</span> ${(item.genuine_score * 100).toFixed(2)}%</div>
      <div><span style="color: var(--text-muted);">MODEL USED:</span> ${escapeHtml(item.model_name)}</div>
      <div><span style="color: var(--text-muted);">LATENCY:</span> ${item.latency_ms.toFixed(2)} ms</div>
      <div><span style="color: var(--text-muted);">FEATURE DIMS:</span> ${item.feature_count}-D MFCC (Mean + Std + Delta)</div>
      <div><span style="color: var(--text-muted);">CLIP DURATION:</span> ${item.audio_duration.toFixed(2)} seconds</div>
    `;

    document.getElementById('historyModal').classList.add('active');
  } catch (err) {
    showToast(`Error inspecting record: ${err.message}`, 'error');
  }
}

function closeModal() {
  document.getElementById('historyModal').classList.remove('active');
}

async function deleteHistoryItem(id) {
  try {
    const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete');
    showToast(`Deleted analysis ${id}`, 'info');
    loadHistory();
  } catch (err) {
    showToast(`Failed to delete record: ${err.message}`, 'error');
  }
}

// --- Research Analytics ---
async function loadResearchAnalytics() {
  try {
    const res = await fetch('/api/analytics');
    const data = await res.json();

    const warningBox = document.getElementById('analyticsUninitializedWarning');
    if (data.status === 'uninitialized') {
      warningBox.style.display = 'block';
      document.getElementById('analyticsComparisonTable').innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 24px;">No trained models found. Train models on ASVspoof 2019 LA to generate benchmarks.</td></tr>';
      return;
    }

    warningBox.style.display = 'none';

    // 1. Comparison Table
    renderAnalyticsTable(data.comparison_table || [], data.selected_model);

    // 2. ROC Curves Chart
    renderAnalyticsRocChart(data.roc_curves || {});

    // 3. Latency Chart
    renderAnalyticsLatencyChart(data.latency_comparison || []);

    // 4. Confusion Matrices
    renderAnalyticsConfusionMatrices(data.confusion_matrices || {});
  } catch (err) {
    console.error('Error loading analytics:', err);
  }
}

function renderAnalyticsTable(rows, winnerName) {
  const tbody = document.getElementById('analyticsComparisonTable');
  if (!tbody) return;

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-muted);">No comparison data available.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const isWinner = r.model_name === winnerName;
    return `
      <tr style="${isWinner ? 'background-color: rgba(198, 255, 0, 0.04);' : ''}">
        <td style="font-weight: 700;">
          ${escapeHtml(r.model_name)}
          ${isWinner ? '<span class="badge badge-winner" style="margin-left: 6px;">WINNER</span>' : ''}
        </td>
        <td style="font-family: var(--font-mono);">${(r.accuracy * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); font-weight: 800; color: var(--accent-lime);">${(r.f1_spoof * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono);">${(r.precision_spoof * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono);">${(r.recall_spoof * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono);">${(r.recall_genuine * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); color: var(--warning-amber); font-weight: 700;">${r.eer_percentage.toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); color: var(--cyan-accent); font-weight: 700;">${r.roc_auc.toFixed(4)}</td>
        <td style="font-family: var(--font-mono);">${r.avg_latency_ms.toFixed(2)} ms</td>
        <td style="font-family: var(--font-mono);">${r.training_time_seconds.toFixed(1)} s</td>
      </tr>
    `;
  }).join('');
}

function renderAnalyticsRocChart(rocData) {
  const canvas = document.getElementById('analyticsRocChart');
  if (!canvas) return;

  if (state.charts.rocChart) {
    state.charts.rocChart.destroy();
  }

  const colors = {
    'Logistic Regression': '#FFBF69',
    'SVM (RBF Kernel)': '#00E5FF',
    'Random Forest': '#C6FF00'
  };

  const datasets = [];

  // Diagonal chance line
  datasets.push({
    label: 'Chance (AUC 0.50)',
    data: [{x: 0, y: 0}, {x: 1, y: 1}],
    borderColor: '#53636C',
    borderDash: [5, 5],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false
  });

  for (const [name, info] of Object.entries(rocData)) {
    const points = (info.points || []).map(p => ({ x: p.fpr, y: p.tpr }));
    datasets.push({
      label: `${name} (AUC ${info.auc.toFixed(3)})`,
      data: points,
      borderColor: colors[name] || '#FFFFFF',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
      fill: false
    });
  }

  state.charts.rocChart = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: 1,
          title: { display: true, text: 'False Positive Rate (FPR)', color: '#84939B' },
          grid: { color: '#24333C' },
          ticks: { color: '#84939B' }
        },
        y: {
          type: 'linear',
          min: 0,
          max: 1,
          title: { display: true, text: 'True Positive Rate (TPR)', color: '#84939B' },
          grid: { color: '#24333C' },
          ticks: { color: '#84939B' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#EAF1F4', font: { family: 'DM Mono', size: 11 } }
        }
      }
    }
  });
}

function renderAnalyticsLatencyChart(latencyData) {
  const canvas = document.getElementById('analyticsLatencyChart');
  if (!canvas) return;

  if (state.charts.latencyChart) {
    state.charts.latencyChart.destroy();
  }

  const labels = latencyData.map(d => d.model_name);
  const data = latencyData.map(d => d.latency_ms);

  state.charts.latencyChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Latency (ms / sample)',
        data: data,
        backgroundColor: ['#FFBF69', '#00E5FF', '#C6FF00'],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Latency (ms)', color: '#84939B' },
          grid: { color: '#24333C' },
          ticks: { color: '#84939B' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#84939B', font: { family: 'DM Mono', size: 11 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderAnalyticsConfusionMatrices(cmDict) {
  const container = document.getElementById('analyticsConfusionMatrices');
  if (!container) return;

  if (Object.keys(cmDict).length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 13px;">No confusion matrix data available.</div>';
    return;
  }

  container.innerHTML = Object.entries(cmDict).map(([name, cm]) => {
    const tn = cm.tn || cm.matrix[0][0];
    const fp = cm.fp || cm.matrix[0][1];
    const fn = cm.fn || cm.matrix[1][0];
    const tp = cm.tp || cm.matrix[1][1];

    return `
      <div class="cm-card">
        <div class="cm-title">${escapeHtml(name)}</div>
        <div class="cm-grid">
          <div></div>
          <div style="color: var(--text-muted); font-size: 10px;">PRED: BONAFIDE</div>
          <div style="color: var(--text-muted); font-size: 10px;">PRED: SPOOF</div>

          <div style="color: var(--text-muted); font-size: 10px; display: flex; align-items: center;">TRUE: BONAFIDE</div>
          <div class="cm-cell correct">
            <span style="font-size: 9px; color: var(--accent-lime);">TN</span>
            <b>${tn}</b>
          </div>
          <div class="cm-cell error">
            <span style="font-size: 9px; color: var(--danger-red);">FP</span>
            <b>${fp}</b>
          </div>

          <div style="color: var(--text-muted); font-size: 10px; display: flex; align-items: center;">TRUE: SPOOF</div>
          <div class="cm-cell error">
            <span style="font-size: 9px; color: var(--danger-red);">FN</span>
            <b>${fn}</b>
          </div>
          <div class="cm-cell correct">
            <span style="font-size: 9px; color: var(--accent-lime);">TP</span>
            <b>${tp}</b>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// --- Module: Model Management ---
function initTrainingControls() {
  const btnStart = document.getElementById('btnStartTraining');
  btnStart.addEventListener('click', async () => {
    const maxTrain = document.getElementById('inputMaxTrain').value.trim();
    const maxDev = document.getElementById('inputMaxDev').value.trim();
    const useCache = document.getElementById('checkUseCache').checked;

    btnStart.disabled = true;
    showToast('Initiating model training benchmark...', 'info');

    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_train: maxTrain ? parseInt(maxTrain) : null,
          max_dev: maxDev ? parseInt(maxDev) : null,
          use_cache: useCache
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to start training');

      showToast('Background training job started!', 'info');
      startTrainingPolling();
    } catch (err) {
      btnStart.disabled = false;
      showToast(`Training error: ${err.message}`, 'error');
    }
  });
}

function startTrainingPolling() {
  clearInterval(state.trainingPollInterval);
  state.trainingPollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/training/status');
      const data = await res.json();

      document.getElementById('trainingStatusText').textContent = data.status.toUpperCase();
      document.getElementById('trainingProgressPercent').textContent = `${data.progress || 0}%`;
      document.getElementById('trainingProgressBar').style.width = `${data.progress || 0}%`;
      document.getElementById('trainingCurrentStepText').textContent = data.current_step || '';

      if (data.logs) {
        const consoleEl = document.getElementById('trainingLogConsole');
        consoleEl.textContent = data.logs;
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }

      if (data.status === 'completed') {
        clearInterval(state.trainingPollInterval);
        document.getElementById('btnStartTraining').disabled = false;
        showToast('Model training completed! Best model updated.', 'info');
        loadModelsView();
        checkSystemHealth();
      } else if (data.status === 'failed') {
        clearInterval(state.trainingPollInterval);
        document.getElementById('btnStartTraining').disabled = false;
        showToast(`Training halted: ${data.error_message}`, 'error');
      }
    } catch (err) {
      console.warn('Polling status error:', err);
    }
  }, 1500);
}

async function loadModelsView() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const container = document.getElementById('modelsCardsList');

    if (!data.models || data.models.length === 0) {
      container.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; grid-column: 1/-1;">No models trained yet. Click "Start Background Training" below.</div>';
      return;
    }

    container.innerHTML = data.models.map(m => {
      const isActive = m.is_active === 1;
      return `
        <div class="panel" style="${isActive ? 'border-color: var(--accent-lime);' : ''}">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div>
              <div style="font-weight: 800; font-size: 15px;">${escapeHtml(m.model_name)}</div>
              <div style="font-size: 11px; color: var(--text-secondary);">${escapeHtml(m.algorithm)}</div>
            </div>
            ${isActive ? '<span class="badge badge-genuine">ACTIVE</span>' : ''}
          </div>

          <div style="font-family: var(--font-mono); font-size: 12px; display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px;">
            <div>Spoof F1: <b style="color: var(--accent-lime);">${(m.f1_spoof * 100).toFixed(2)}%</b></div>
            <div>Accuracy: <b>${(m.accuracy * 100).toFixed(2)}%</b></div>
            <div>EER: <b>${(m.eer * 100).toFixed(2)}%</b></div>
            <div>Latency: <b>${m.avg_latency_ms.toFixed(2)} ms</b></div>
          </div>

          ${!isActive ? `<button class="btn-secondary" style="width: 100%; font-size: 12px;" onclick="setActiveModel('${escapeHtml(m.model_name)}')">Set as Active Model</button>` : ''}
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error('Error loading models view:', err);
  }
}

async function setActiveModel(modelName) {
  try {
    const res = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName })
    });
    if (!res.ok) throw new Error('Failed to set model');
    showToast(`Active model changed to: ${modelName}`, 'info');
    loadModelsView();
    checkSystemHealth();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// --- Module: Dataset Management ---
async function loadDatasetInfo() {
  try {
    const res = await fetch('/api/dataset/info');
    const data = await res.json();

    document.getElementById('datasetRootPathDisplay').textContent = data.root_path;

    const grid = document.getElementById('datasetPartitionsGrid');
    if (!data.partitions) {
      grid.innerHTML = '<div style="color: var(--text-muted);">No partition details.</div>';
      return;
    }

    grid.innerHTML = Object.entries(data.partitions).map(([split, p]) => {
      const isFound = p.protocol_found;
      return `
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="font-weight: 700; font-size: 14px; text-transform: uppercase;">${split} Partition</div>
            <span class="badge ${isFound ? 'badge-genuine' : 'badge-synthetic'}">${isFound ? 'READY' : 'MISSING'}</span>
          </div>
          <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 4px;">
            <div>Protocol: <b>${escapeHtml(p.protocol_file)}</b></div>
            <div>Sample Count: <b>${p.sample_count.toLocaleString()}</b></div>
            <div>Bonafide: <b style="color: var(--accent-lime);">${p.bonafide_count.toLocaleString()}</b></div>
            <div>Spoof: <b style="color: var(--danger-red);">${p.spoof_count.toLocaleString()}</b></div>
            <div>Feature Cache: <b>${p.cache_exists ? 'Cached (.npz)' : 'Uncached'}</b></div>
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error('Error loading dataset info:', err);
  }
}

// --- Settings ---
function initSettings() {
  const slider = document.getElementById('settingThreshold');
  const val = document.getElementById('settingThresholdVal');
  if (slider && val) {
    slider.addEventListener('input', () => {
      val.textContent = parseFloat(slider.value).toFixed(2);
    });
  }
}

// --- Utilities ---
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
