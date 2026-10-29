/* ============================================
   CrowdGuard — Application Logic
   Pipeline orchestration, upload, animations
   ============================================ */

// ===== Utility Functions =====
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Everything from the server — and especially from Gemma — is untrusted text.
// Escape it before it goes anywhere near innerHTML.
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fmtScore(value, digits = 4) {
  return typeof value === 'number' && isFinite(value) ? value.toFixed(digits) : '--';
}

function shortHash(hash) {
  return hash ? `${hash.slice(0, 12)}…${hash.slice(-6)}` : '--';
}

const API_BASE = window.location.origin;

async function postJson(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  try {
    return await resp.json();
  } catch (e) {
    throw new Error(`Server returned HTTP ${resp.status}`);
  }
}

// Compute gate for Stage 2: only the highest-scoring VAD events go to Gemma.
const MAX_GEMMA_EVENTS = 3;

// Camera → location registry (static until cameras come from a config/API).
const CAMERA_LOCATIONS = {
  'CAM-01': 'Sector 4 (North Gate)'
};

const DEPARTMENTS = ['Police', 'Hospital', 'Fire Dept.'];

function getDepartmentForAnomaly(class_name) {
  const c = (class_name || "").toLowerCase();
  if (c.includes("arson") || c.includes("explosion") || c.includes("fire")) {
    return "Fire Dept.";
  } else if (c.includes("accident") || c.includes("road")) {
    return "Hospital";
  } else {
    return "Police";
  }
}

// ===== State =====
const state = {
  videoFile: null,
  isProcessing: false,
  pipelineResults: {},
  sessionId: null,
  cameraId: 'CAM-01',
  pipelineAborted: false,
  pipelineFailed: false,
  abortReason: ''
};

let startButtonHtml = '';

// ===== Navigation =====
function initNavigation() {
  const navbar = document.getElementById('navbar');
  const navLinks = document.querySelectorAll('.nav-links a');

  // Scroll effect
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  // Active link tracking
  const sections = document.querySelectorAll('section[id]');
  const observerOptions = { rootMargin: '-20% 0px -80% 0px' };

  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (activeLink) activeLink.classList.add('active');
      }
    });
  }, observerOptions);

  sections.forEach(section => sectionObserver.observe(section));

  // Smooth scroll for nav links
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = link.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // Mobile toggle
  const mobileToggle = document.getElementById('mobile-toggle');
  const navLinksContainer = document.getElementById('nav-links');
  if (mobileToggle) {
    mobileToggle.addEventListener('click', () => {
      navLinksContainer.style.display = navLinksContainer.style.display === 'flex' ? 'none' : 'flex';
    });
  }
}

// ===== Scroll Reveal Animations =====
function initRevealAnimations() {
  const revealElements = document.querySelectorAll('[data-reveal]');

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  revealElements.forEach(el => revealObserver.observe(el));
}

// ===== Pipeline Tabs =====
function initPipelineTabs() {
  const tabs = document.querySelectorAll('.pipeline-tab');
  const tabMap = {
    'vad': { card: 'tab-vad', visual: 'visual-vad' },
    'gemma': { card: 'tab-gemma', visual: 'visual-gemma' },
    'yolo': { card: 'tab-yolo', visual: 'visual-yolo' },
    'reid': { card: 'tab-reid', visual: 'visual-reid' },
    'forensic': { card: 'tab-forensic', visual: 'visual-forensic' }
  };

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;

      // Update active tab
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      // Update cards and visuals
      document.querySelectorAll('.feature-card').forEach(c => c.classList.remove('active'));
      document.querySelectorAll('.feature-visual').forEach(v => v.classList.remove('active'));

      const mapping = tabMap[tabId];
      if (mapping) {
        document.getElementById(mapping.card).classList.add('active');
        document.getElementById(mapping.visual).classList.add('active');
      }
    });
  });
}

// ===== File Upload =====
function initUpload() {
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const videoPreview = document.getElementById('video-preview-container');
  const videoPlayer = document.getElementById('video-player');
  const fileName = document.getElementById('file-name');
  const fileSize = document.getElementById('file-size');
  const startWrapper = document.getElementById('start-analysis-wrapper');
  const removeBtn = document.getElementById('remove-file-btn');

  // Click to upload
  uploadZone.addEventListener('click', () => fileInput.click());

  // Drag and drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleFileSelect(file);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelect(file);
  });

  // Remove file
  removeBtn.addEventListener('click', () => {
    resetUpload();
  });

  function handleFileSelect(file) {
    // Swapping the video mid-run would leave async stages writing into a reset page.
    if (state.isProcessing) return;
    resetPipeline();
    state.videoFile = file;

    // Update UI
    uploadZone.classList.add('has-file');
    uploadZone.querySelector('h3').textContent = '✓ Video selected';
    uploadZone.querySelector('p').textContent = file.name;

    // Show video preview
    if (videoPlayer.src) URL.revokeObjectURL(videoPlayer.src);
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    videoPreview.classList.add('visible');
    startWrapper.classList.add('visible');

    // Scroll to preview
    setTimeout(() => {
      videoPreview.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
  }

  function resetUpload() {
    if (state.isProcessing) return;
    state.videoFile = null;

    uploadZone.classList.remove('has-file');
    uploadZone.querySelector('h3').textContent = 'Drop your video clip here';
    uploadZone.querySelector('p').textContent = 'or click to browse files';

    videoPreview.classList.remove('visible');
    startWrapper.classList.remove('visible');
    if (videoPlayer.src) URL.revokeObjectURL(videoPlayer.src);
    videoPlayer.removeAttribute('src');
    fileInput.value = '';

    // Reset pipeline
    resetPipeline();
  }
}

// ===== Pipeline Orchestration =====
function skipStages(from, to, reason) {
  for (let i = from; i <= to; i++) {
    skipStageCard(i);
    updateStageStatus(i, 'skipped');
    const resultsEl = document.getElementById(`stage-${i}-results`);
    resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-tertiary); font-size: 14px;">⏭ Skipped — ${escapeHtml(reason)}</div>`;
  }
}

function initPipeline() {
  const startBtn = document.getElementById('start-analysis-btn');
  startButtonHtml = startBtn.innerHTML;

  const finish = (label) => {
    startBtn.innerHTML = label;
    state.isProcessing = false;
  };

  startBtn.addEventListener('click', async () => {
    if (state.isProcessing || !state.videoFile) return;
    state.isProcessing = true;
    state.pipelineAborted = false;
    state.pipelineFailed = false;
    state.abortReason = '';
    const cameraInput = document.getElementById('camera-id-input');
    state.cameraId = (cameraInput && cameraInput.value.trim()) || 'CAM-01';
    startBtn.disabled = true;
    startBtn.innerHTML = '<div class="spinner"></div> Processing...';

    // Show pipeline section
    const pipelineSection = document.getElementById('pipeline-processing');
    pipelineSection.classList.add('visible');

    // Scroll to pipeline
    setTimeout(() => {
      pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

    // Stage 1: VAD
    await runStage1_VAD();

    // If no anomaly detected, skip all remaining stages
    if (state.pipelineAborted) {
      skipStages(2, 7, state.abortReason);
      finish(state.pipelineFailed ? '✕ Pipeline Error — see Stage 1' : '✓ Analysis Complete — No Anomaly');
      return;
    }

    // Stage 2: Gemma
    await runStage2_Gemma();

    // If Gemma rejected the anomaly, skip stages 3-7
    if (state.pipelineAborted) {
      skipStages(3, 7, state.abortReason);
      if (!state.pipelineFailed) showComparison();
      finish(state.pipelineFailed ? '✕ Pipeline Error — see Stage 2' : '✓ Analysis Complete — Anomaly Rejected by Gemma');
      return;
    }

    // Stages 3-5 only run when Gemma says a person is involved
    await runStages3to5_Tracking();
    await runStage6_ForensicLog();
    await runStage7_AlertDashboard();

    showComparison();
    finish('✓ Analysis Complete');
  });
}

function updateStageStatus(stageNum, status) {
  const statusEl = document.getElementById(`stage-${stageNum}-status`);
  statusEl.className = `stage-status ${status}`;

  const labels = {
    pending: 'Pending',
    running: 'Processing...',
    completed: 'Completed',
    error: 'Error',
    skipped: 'Skipped'
  };

  statusEl.querySelector('span').textContent = labels[status];
}

function showStageCard(stageNum) {
  const card = document.getElementById(`stage-${stageNum}`);
  card.classList.add('visible', 'active');
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function completeStageCard(stageNum) {
  const card = document.getElementById(`stage-${stageNum}`);
  card.classList.remove('active');
  card.classList.add('completed');
}

function skipStageCard(stageNum) {
  const card = document.getElementById(`stage-${stageNum}`);
  card.classList.add('visible', 'skipped');
  card.classList.remove('active', 'completed');
}

function renderStageError(stageNum, title, message, hint) {
  document.getElementById(`stage-${stageNum}-results`).innerHTML = `
    <div style="padding: 20px; background: #FEE2E2; border-radius: 12px; text-align: center;">
      <div style="font-size: 32px; margin-bottom: 8px;">❌</div>
      <strong style="color: #991B1B;">${escapeHtml(title)}</strong>
      <p style="font-size: 13px; color: #991B1B; margin-top: 4px;">${escapeHtml(message)}</p>
      ${hint ? `<p style="font-size: 11px; color: #991B1B; margin-top: 8px;">${escapeHtml(hint)}</p>` : ''}
    </div>
  `;
  updateStageStatus(stageNum, 'error');
}

// ===== Stage 1: VAD =====
async function runStage1_VAD() {
  showStageCard(1);
  updateStageStatus(1, 'running');

  const resultsEl = document.getElementById('stage-1-results');
  resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Uploading video and running VadCLIP inference...<br><span style="font-size: 12px; color: var(--color-text-tertiary);">This may take 30-90 seconds depending on video length</span></div>`;

  try {
    // Step 1: Upload video
    const formData = new FormData();
    formData.append('video', state.videoFile);
    formData.append('camera_id', state.cameraId);

    const uploadResp = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });
    let uploadData;
    try {
      uploadData = await uploadResp.json();
    } catch (e) {
      throw new Error(`Upload failed (HTTP ${uploadResp.status})`);
    }

    if (uploadData.status !== 'success') {
      throw new Error(uploadData.message || 'Upload failed');
    }

    state.sessionId = uploadData.session_id;
    state.cameraId = uploadData.camera_id || state.cameraId;

    resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Video uploaded ✓ — Running anomaly detection model...<br><span style="font-size: 12px; color: var(--color-text-tertiary);">Extracting CLIP features → VadCLIP inference → Grouping anomalous intervals</span></div>`;

    // Step 2: Run VAD
    const vadData = await postJson('/api/pipeline/vad', { session_id: state.sessionId });

    if (vadData.status !== 'success') {
      throw new Error(vadData.message || 'VAD pipeline failed');
    }

    // Store results
    const maxScore = vadData.summary.overall_max_score;
    state.pipelineResults.vadScore = maxScore;
    state.pipelineResults.vadLatency = vadData.latency_ms;
    state.pipelineResults.vadEvents = vadData.events;
    state.pipelineResults.vadThreshold = vadData.summary.threshold;
    state.pipelineResults.vadSnippetScores = vadData.snippet_scores;

    const anomalyDetected = vadData.anomaly_detected;
    const scoreClass = maxScore > 0.5 ? 'high' : maxScore > 0.1 ? 'medium' : 'low';
    const threshold = vadData.summary.threshold;

    if (anomalyDetected) {
      // Anomaly detected — show results and proceed
      const topEvent = [...vadData.events].sort((a, b) => b.max_score - a.max_score)[0];

      resultsEl.innerHTML = `
        <div class="result-grid">
          <div class="result-item">
            <div class="result-label">Max Anomaly Score</div>
            <div class="result-value score-${scoreClass}">${fmtScore(maxScore)}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${vadData.latency_ms}ms</div>
          </div>
          <div class="result-item">
            <div class="result-label">Threshold</div>
            <div class="result-value">${threshold}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Events Found</div>
            <div class="result-value" style="color: var(--color-danger);">${vadData.events.length} anomal${vadData.events.length > 1 ? 'ies' : 'y'}</div>
          </div>
        </div>
        <div class="score-bar" style="margin-top: 16px;">
          <div class="score-bar-fill ${scoreClass}" style="width: ${Math.min(maxScore * 100, 100)}%;" data-score="${maxScore}"></div>
        </div>
        ${vadData.events.map(evt => `
          <div style="margin-top: 12px; padding: 12px; background: var(--color-bg-subtle); border-radius: 8px; border-left: 3px solid var(--color-danger);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <strong style="font-size: 14px;">Event ${evt.event_id}: ${escapeHtml(evt.predicted_class)}</strong>
              <span style="padding: 3px 10px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 11px; font-weight: 600;">Score: ${fmtScore(evt.max_score)}</span>
            </div>
            <span style="font-size: 12px; color: var(--color-text-tertiary);">Time: ${evt.start_time_sec}s → ${evt.end_time_sec}s</span>
          </div>
        `).join('')}
        ${topEvent.collage_url ? `
          <div style="margin-top: 16px; text-align: center;">
            <p style="font-size: 12px; color: var(--color-text-tertiary); margin-bottom: 8px;">VAD frame collage — Event ${topEvent.event_id} (highest score)</p>
            <img src="${escapeHtml(topEvent.collage_url)}" alt="Anomaly Event Collage" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--color-border);" />
          </div>
        ` : ''}
        <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
          <span style="padding: 4px 12px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 12px; font-weight: 600;">⚠ Anomaly detected: ${escapeHtml(topEvent.predicted_class)}</span>
          <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">→ Forwarding top ${Math.min(vadData.events.length, MAX_GEMMA_EVENTS)} to Gemma for confirmation</span>
        </div>
      `;
    } else {
      // No anomaly — show clean result and abort pipeline
      resultsEl.innerHTML = `
        <div class="result-grid">
          <div class="result-item">
            <div class="result-label">Max Anomaly Score</div>
            <div class="result-value" style="color: var(--color-success);">${fmtScore(maxScore)}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${vadData.latency_ms}ms</div>
          </div>
          <div class="result-item">
            <div class="result-label">Threshold</div>
            <div class="result-value">${threshold}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Decision</div>
            <div class="result-value" style="color: var(--color-success);">✓ No anomaly</div>
          </div>
        </div>
        <div class="score-bar" style="margin-top: 16px;">
          <div class="score-bar-fill low" style="width: ${Math.min(maxScore * 100, 100)}%;" data-score="${maxScore}"></div>
        </div>
        <div style="margin-top: 16px; padding: 20px; text-align: center; background: #D1FAE5; border-radius: 12px;">
          <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
          <strong style="color: #065F46;">No Anomaly Detected</strong>
          <p style="font-size: 13px; color: #065F46; margin-top: 4px;">All anomaly scores are below threshold (${threshold}). Video appears normal. Remaining pipeline stages skipped.</p>
        </div>
      `;

      state.pipelineAborted = true;
      state.abortReason = 'No anomaly detected by VAD';
    }

    updateStageStatus(1, 'completed');
    completeStageCard(1);
    await sleep(400);

  } catch (err) {
    console.error('Stage 1 VAD error:', err);
    renderStageError(1, 'VAD Pipeline Error', err.message,
      'Check that the server is running and model weights are available.');
    state.pipelineAborted = true;
    state.pipelineFailed = true;
    state.abortReason = 'VAD pipeline error';
  }
}

// ===== Stage 2: Gemma =====
async function runStage2_Gemma() {
  showStageCard(2);
  updateStageStatus(2, 'running');

  const r = state.pipelineResults;
  const resultsEl = document.getElementById('stage-2-results');
  const candidates = [...r.vadEvents]
    .sort((a, b) => b.max_score - a.max_score)
    .slice(0, MAX_GEMMA_EVENTS);

  try {
    const verdicts = [];
    for (let i = 0; i < candidates.length; i++) {
      const evt = candidates[i];
      resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Sending frame grid to Gemma via Kaggle VM... (event ${i + 1} of ${candidates.length})<br><span style="font-size: 12px; color: var(--color-text-tertiary);">Creating 3×3 frame grid → Uploading to ngrok endpoint → Awaiting inference</span></div>`;
      try {
        const data = await postJson('/api/pipeline/gemma', {
          session_id: state.sessionId,
          event_index: evt.event_id - 1
        });
        if (data.status !== 'success') throw new Error(data.message || 'Gemma inference failed');
        verdicts.push({ event: evt, ok: true, data });
      } catch (err) {
        console.error(`Stage 2 Gemma error (event ${evt.event_id}):`, err);
        verdicts.push({ event: evt, ok: false, error: err.message });
      }
    }
    r.gemmaVerdicts = verdicts;

    const answered = verdicts.filter(v => v.ok);
    if (answered.length === 0) {
      throw new Error(verdicts[0] ? verdicts[0].error : 'No events to confirm');
    }

    // The primary event is the most severe confirmed one; otherwise show the top answered one.
    const confirmedVerdicts = answered
      .filter(v => v.data.anomaly_confirmed)
      .sort((a, b) => b.data.severity - a.data.severity);
    const primary = confirmedVerdicts[0] || answered[0];
    const confirmed = confirmedVerdicts.length > 0;

    const gemmaResult = primary.data.gemma_result;
    const confidence = gemmaResult.confidence || 0.0;
    const personInvolved = !!gemmaResult.person_involved;
    const anomalyType = gemmaResult.anomaly_type || 'none';

    r.primaryEventIndex = primary.event.event_id - 1;
    r.primaryVadScore = primary.data.vad_score;
    r.gemmaResult = gemmaResult;
    r.gemmaConfidence = confidence;
    r.gemmaLatency = answered.reduce((sum, v) => sum + (v.data.latency_ms || 0), 0);
    r.gemmaCalls = answered.length;
    r.personInvolved = personInvolved;
    r.anomalyType = anomalyType;
    r.vadVerdict = primary.data.vad_verdict;
    r.severity = primary.data.severity;

    // Format the result as a typed-out JSON block
    const resultJson = JSON.stringify(gemmaResult, null, 2);

    // Show the result with typing animation
    resultsEl.innerHTML = `
      <div class="reasoning-text" id="gemma-reasoning-text" style="font-family: monospace; white-space: pre-wrap; font-size: 13px; background: #282c34; color: #abb2bf; padding: 16px; border-radius: 8px;">
        <strong style="color: #61afef;">🧠 Gemma Semantic Analysis Output — Event ${primary.event.event_id}</strong><br><br><span id="gemma-typing" style="color: #98c379;"></span><span class="typing-cursor"></span>
      </div>
    `;

    // Type out the JSON response character by character
    const typingEl = document.getElementById('gemma-typing');
    for (const char of resultJson) {
      typingEl.textContent += char;
      await sleep(randomBetween(1, 5));
    }

    // Remove cursor after typing
    const cursor = resultsEl.querySelector('.typing-cursor');
    if (cursor) cursor.remove();

    const verdictRows = verdicts.map(v => {
      let label, color;
      if (!v.ok) {
        label = `⚠ Error: ${v.error}`;
        color = 'var(--color-warning)';
      } else if (v.data.gemma_result.needs_review) {
        label = '⚠ Unparseable output — needs human review';
        color = 'var(--color-warning)';
      } else if (v.data.anomaly_confirmed) {
        label = `Confirmed (${v.data.gemma_result.confidence})`;
        color = 'var(--color-danger)';
      } else {
        label = 'Rejected';
        color = 'var(--color-success)';
      }
      return `
        <div style="padding: 8px 12px; background: var(--color-bg-subtle); border-radius: 8px; font-size: 13px; display: flex; justify-content: space-between; gap: 12px;">
          <span>Event ${v.event.event_id}: ${escapeHtml(v.event.predicted_class)} (VAD ${fmtScore(v.event.max_score)})</span>
          <span style="color: ${color}; font-weight: 600;">${escapeHtml(label)}</span>
        </div>`;
    }).join('');

    resultsEl.innerHTML += `
      ${primary.data.grid_url ? `
        <div style="margin-top: 16px; text-align: center;">
          <p style="font-size: 12px; color: var(--color-text-tertiary); margin-bottom: 8px;">Exact 3×3 grid Gemma analysed (${primary.data.event_time_range.start_sec}s → ${primary.data.event_time_range.end_sec}s)</p>
          <img src="${escapeHtml(primary.data.grid_url)}" alt="Frame grid sent to Gemma" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--color-border);" />
        </div>` : ''}
      <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 16px;">${verdictRows}</div>
    `;

    if (confirmed) {
      // Anomaly confirmed by Gemma
      resultsEl.innerHTML += `
        <div class="result-grid" style="margin-top: 16px;">
          <div class="result-item">
            <div class="result-label">Confirmation</div>
            <div class="result-value" style="color: var(--color-danger);">Anomaly Confirmed ✓</div>
          </div>
          <div class="result-item">
            <div class="result-label">Confidence</div>
            <div class="result-value">${escapeHtml(confidence)}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Person Involved</div>
            <div class="result-value" style="color: ${personInvolved ? 'var(--color-success)' : 'var(--color-text-secondary)'};">${personInvolved ? 'Yes → Stage 3' : 'No → skip tracking'}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${primary.data.latency_ms}ms</div>
          </div>
        </div>
        <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
          <span style="padding: 4px 12px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 12px; font-weight: 600;">⚠ Anomaly confirmed: ${escapeHtml(anomalyType)}</span>
          <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">VAD verdict: ${escapeHtml(primary.data.vad_verdict)}</span>
        </div>
      `;
    } else {
      // Anomaly rejected by Gemma
      resultsEl.innerHTML += `
        <div class="result-grid" style="margin-top: 16px;">
          <div class="result-item">
            <div class="result-label">Confirmation</div>
            <div class="result-value" style="color: var(--color-success);">Anomaly Rejected</div>
          </div>
          <div class="result-item">
            <div class="result-label">Confidence</div>
            <div class="result-value">${escapeHtml(confidence)}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Person Involved</div>
            <div class="result-value">${personInvolved ? 'Yes' : 'No'}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${r.gemmaLatency}ms</div>
          </div>
        </div>
        <div style="margin-top: 16px; padding: 20px; text-align: center; background: #D1FAE5; border-radius: 12px;">
          <div style="font-size: 32px; margin-bottom: 8px;">🟢</div>
          <strong style="color: #065F46;">Anomaly Rejected by Gemma</strong>
          <p style="font-size: 13px; color: #065F46; margin-top: 4px;">Gemma's semantic analysis did not confirm any of the ${answered.length} event(s) flagged by VAD. False positives filtered. Remaining stages skipped.</p>
        </div>
      `;

      state.pipelineAborted = true;
      state.abortReason = 'Anomaly rejected by Gemma semantic confirmation';
    }

    updateStageStatus(2, 'completed');
    completeStageCard(2);
    await sleep(400);

  } catch (err) {
    console.error('Stage 2 Gemma error:', err);
    renderStageError(2, 'Gemma Inference Error', err.message,
      'Check that the Kaggle VM is running and GEMMA_SERVER_URL points at its current ngrok URL.');
    state.pipelineAborted = true;
    state.pipelineFailed = true;
    state.abortReason = 'Gemma inference error';
  }
}

// ===== Stages 3-5: YOLO → OSNet → Cross-Camera (one backend call) =====
async function runStages3to5_Tracking() {
  const r = state.pipelineResults;
  r.tracking = null;

  if (!r.personInvolved) {
    skipStages(3, 5, 'Gemma reported no person involved — person tracking not needed');
    return;
  }

  showStageCard(3);
  updateStageStatus(3, 'running');
  const stage3El = document.getElementById('stage-3-results');
  stage3El.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Running YOLOv8n on frames from the confirmed event...</div>`;

  let data;
  try {
    data = await postJson('/api/pipeline/track', {
      session_id: state.sessionId,
      event_index: r.primaryEventIndex
    });
  } catch (err) {
    data = { status: 'error', message: err.message };
  }

  if (data.status === 'unavailable' || data.status === 'skipped') {
    skipStages(3, 5, data.message || 'Tracking unavailable');
    return;
  }
  if (data.status !== 'success') {
    console.error('Stage 3 tracking error:', data);
    renderStageError(3, 'Tracking Error', data.message || 'Tracking failed');
    skipStages(4, 5, 'Person localization failed');
    return;
  }
  r.tracking = data;

  // --- Stage 3: YOLO ---
  const numDetections = data.detections.length;
  stage3El.innerHTML = `
    <div class="yolo-canvas-container" style="text-align: center; border-radius: 12px; overflow: hidden; border: 1px solid var(--color-border);">
      <img src="${escapeHtml(data.annotated_url)}" alt="YOLO Detections" style="width: 100%; display: block;" />
    </div>
    <div class="result-grid" style="margin-top: 16px;">
      <div class="result-item">
        <div class="result-label">Persons Detected</div>
        <div class="result-value">${numDetections}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Detection Latency</div>
        <div class="result-value">${data.yolo_latency_ms}ms/frame</div>
      </div>
      <div class="result-item">
        <div class="result-label">Keyframe</div>
        <div class="result-value" style="font-size: 14px;">${data.keyframe_time_sec}s (best of ${data.frames_scanned})</div>
      </div>
      <div class="result-item">
        <div class="result-label">Next</div>
        <div class="result-value" style="color: var(--color-indigo); font-size: 14px;">${numDetections ? 'Crop → Re-ID' : 'Nothing to embed'}</div>
      </div>
    </div>
  `;
  updateStageStatus(3, 'completed');
  completeStageCard(3);
  await sleep(400);

  if (numDetections === 0) {
    skipStages(4, 5, 'YOLO found no person in the sampled frames');
    return;
  }

  // --- Stage 4: OSNet Re-ID ---
  showStageCard(4);
  updateStageStatus(4, 'running');
  const preview = data.embedding_preview || [];
  const maxAbs = Math.max(...preview.map(Math.abs), 1e-6);
  const embeddingHTML = preview.map(val => {
    const normalized = (val / maxAbs + 1) / 2;
    const hue = Math.round(normalized * 240);
    return `<div class="embedding-cell" style="background: hsl(${hue}, 70%, 55%); height: ${Math.round(10 + normalized * 20)}px;"></div>`;
  }).join('');
  const imagenetOnly = data.reid_weights === 'imagenet';

  document.getElementById('stage-4-results').innerHTML = `
    <div style="margin-bottom: 16px;">
      <p style="font-size: 13px; color: var(--color-text-secondary); margin-bottom: 8px;">Person 0 embedding — first ${preview.length} of ${data.embedding_dim} dimensions</p>
      <div class="embedding-viz" style="flex-direction: column;">
        <div class="embedding-vector" style="gap: 2px; justify-content: center;">
          ${embeddingHTML}
        </div>
      </div>
    </div>
    <div class="result-grid">
      <div class="result-item">
        <div class="result-label">Embedding Dim</div>
        <div class="result-value">${data.embedding_dim}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Extraction Time</div>
        <div class="result-value">${data.reid_latency_ms}ms (${numDetections} crop${numDetections > 1 ? 's' : ''})</div>
      </div>
      <div class="result-item">
        <div class="result-label">Model</div>
        <div class="result-value" style="font-size: 14px;">OSNet x1.0</div>
      </div>
      <div class="result-item">
        <div class="result-label">Weights</div>
        <div class="result-value" style="font-size: 14px; color: ${imagenetOnly ? 'var(--color-warning)' : 'inherit'};">${escapeHtml(data.reid_weights)}</div>
      </div>
    </div>
    ${imagenetOnly ? `<p style="font-size: 12px; color: var(--color-warning); margin-top: 12px;">⚠ No Re-ID checkpoint configured (OSNET_WEIGHTS) — ImageNet features are not trained for person matching, so similarity scores below are unreliable.</p>` : ''}
  `;
  updateStageStatus(4, 'completed');
  completeStageCard(4);
  await sleep(400);

  // --- Stage 5: Cross-camera matching ---
  showStageCard(5);
  updateStageStatus(5, 'running');
  const cc = data.cross_camera;
  const scored = cc.matches.filter(m => m.similarity !== null);
  const best = scored.sort((a, b) => b.similarity - a.similarity)[0];
  const otherCameras = cc.cameras_in_gallery.filter(c => c !== cc.camera_id);
  r.crossCameraMatch = cc.any_match;
  r.similarity = best ? best.similarity : null;
  r.matchedCamera = best && best.matched ? best.matched_camera : null;

  document.getElementById('stage-5-results').innerHTML = `
    <div class="result-grid" style="margin-top: 16px;">
      <div class="result-item">
        <div class="result-label">Match Status</div>
        <div class="result-value" style="color: ${cc.any_match ? 'var(--color-success)' : 'var(--color-info)'};">${cc.any_match ? `✓ Seen on ${escapeHtml(r.matchedCamera)}` : (otherCameras.length ? 'New Individual' : 'First camera — stored')}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Best Cosine Similarity</div>
        <div class="result-value">${best ? fmtScore(best.similarity, 2) : '--'}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Threshold</div>
        <div class="result-value">${fmtScore(cc.threshold, 2)}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Other Cameras in Gallery</div>
        <div class="result-value">${otherCameras.length}</div>
      </div>
    </div>
    ${otherCameras.length ? '' : `<p style="font-size: 12px; color: var(--color-text-tertiary); margin-top: 12px;">No other camera has reported people yet. Upload a clip from a different Camera ID to test cross-camera matching.</p>`}
  `;
  updateStageStatus(5, 'completed');
  completeStageCard(5);
  await sleep(400);
}

// ===== Stage 6: Forensic Log =====
async function runStage6_ForensicLog() {
  showStageCard(6);
  updateStageStatus(6, 'running');

  const r = state.pipelineResults;
  r.logEntry = null;

  try {
    const data = await postJson('/api/pipeline/log', {
      session_id: state.sessionId,
      event_index: r.primaryEventIndex
    });
    if (data.status !== 'success') throw new Error(data.message || 'Forensic logging failed');

    const entry = data.entry;
    r.logEntry = entry;
    r.severity = entry.payload.severity_score;

    // Escape first (quotes kept so the highlighter can still find keys/strings).
    const jsonStr = JSON.stringify(entry, null, 2)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const highlighted = jsonStr
      .replace(/"([^"]+)":/g, '<span class="log-key">"$1"</span>:')
      .replace(/: "([^"]*)"/g, ': <span class="log-string">"$1"</span>')
      .replace(/: (-?\d+\.?\d*(?:e[-+]?\d+)?)/gi, ': <span class="log-number">$1</span>')
      .replace(/: (true|false|null)/g, ': <span class="log-number">$1</span>');

    document.getElementById('stage-6-results').innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
        <span style="padding: 4px 12px; background: #FEF3C7; color: #D97706; border-radius: 999px; font-size: 12px; font-weight: 600;">🔗 SHA-256 hash-chained</span>
        <span style="padding: 4px 12px; background: #D1FAE5; color: #059669; border-radius: 999px; font-size: 12px; font-weight: 600;">📎 Append-only JSONL</span>
        <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">🔐 Verify: /api/forensic/verify</span>
      </div>
      <div class="forensic-log"><pre style="margin: 0; white-space: pre-wrap; word-break: break-all; font-family: inherit;">${highlighted}</pre></div>
      <div class="result-grid" style="margin-top: 16px;">
        <div class="result-item">
          <div class="result-label">Entry #</div>
          <div class="result-value" style="font-size: 13px;">${entry.seq}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Prev Hash</div>
          <div class="result-value" style="font-size: 13px; color: var(--color-warning);" title="${escapeHtml(entry.prev_hash)}">${shortHash(entry.prev_hash)}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Current Hash</div>
          <div class="result-value" style="font-size: 13px; color: var(--color-warning);" title="${escapeHtml(entry.hash)}">${shortHash(entry.hash)}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Severity Score</div>
          <div class="result-value" style="color: var(--color-danger);">${fmtScore(r.severity, 4)}</div>
        </div>
      </div>
    `;

    updateStageStatus(6, 'completed');
    completeStageCard(6);
  } catch (err) {
    console.error('Stage 6 forensic log error:', err);
    renderStageError(6, 'Forensic Log Error', err.message,
      'Operator decisions cannot be recorded until the event is logged.');
  }
  await sleep(400);
}

// ===== Stage 7: Alert Dashboard =====
async function runStage7_AlertDashboard() {
  showStageCard(7);
  updateStageStatus(7, 'running');

  const r = state.pipelineResults;
  const severity = typeof r.severity === 'number' ? r.severity : 0;
  const tracking = r.tracking;

  const latencyRows = [
    ['VAD (VadCLIP)', r.vadLatency, '#3B82F6'],
    [`Gemma (${r.gemmaCalls} call${r.gemmaCalls > 1 ? 's' : ''})`, r.gemmaLatency, '#E67E22'],
    ['YOLOv8n (per frame)', tracking ? tracking.yolo_latency_ms : null, '#8B5CF6'],
    ['OSNet Re-ID', tracking ? tracking.reid_latency_ms : null, '#EC4899']
  ];
  const totalLatency = latencyRows.reduce((sum, row) => sum + (row[1] || 0), 0) || 1;

  const severityColor = severity > 0.7 ? 'var(--color-danger)' : severity > 0.4 ? 'var(--color-warning)' : 'var(--color-success)';
  const severityLabel = severity > 0.7 ? 'High Severity' : severity > 0.4 ? 'Medium Severity' : 'Low Severity';

  let trackingSummary = 'Person tracking was skipped for this event.';
  if (tracking) {
    trackingSummary = `YOLO detected ${tracking.detections.length} person(s) at ${tracking.keyframe_time_sec}s. `;
    if (r.crossCameraMatch) {
      trackingSummary += `Cross-camera match on ${r.matchedCamera} (similarity ${fmtScore(r.similarity, 2)}).`;
    } else if (r.similarity !== null && r.similarity !== undefined) {
      trackingSummary += `No cross-camera match (best similarity ${fmtScore(r.similarity, 2)}).`;
    } else {
      trackingSummary += 'No other cameras in the gallery to match against yet.';
    }
  }

  document.getElementById('stage-7-results').innerHTML = `
    <div class="alert-dashboard">
      <div class="severity-gauge">
        <div class="severity-circle" style="background: ${severityColor}; color: white; box-shadow: 0 0 30px ${severityColor}40;">
          ${severity.toFixed(2)}
        </div>
        <div class="severity-info">
          <h4>${severityLabel}</h4>
          <p>VAD (${fmtScore(r.primaryVadScore)}) × Gemma (${escapeHtml(r.gemmaConfidence)}) = ${fmtScore(severity)}</p>
          <p style="font-size: 12px; color: var(--color-text-tertiary); margin-top: 4px;">Not binary — weighted severity score</p>
        </div>
      </div>

      <div style="background: var(--color-bg-subtle); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="font-family: var(--font-display); font-size: 16px; margin-bottom: 12px;">📊 Pipeline Latency Breakdown</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${latencyRows.map(([label, ms, color]) => `
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="width: 140px; font-size: 13px; color: var(--color-text-secondary);">${escapeHtml(label)}</span>
            <div style="flex: 1; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
              <div style="width: ${((ms || 0) / totalLatency) * 100}%; height: 100%; background: ${color}; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; font-weight: 600; width: 70px; text-align: right;">${ms === null || ms === undefined ? 'skipped' : `${ms}ms`}</span>
          </div>`).join('')}
          <div style="display: flex; align-items: center; gap: 12px; padding-top: 8px; border-top: 1px solid var(--color-border);">
            <span style="width: 140px; font-size: 13px; font-weight: 600;">Total</span>
            <div style="flex: 1;"></div>
            <span style="font-size: 14px; font-weight: 700; width: 70px; text-align: right;">${totalLatency}ms</span>
          </div>
        </div>
      </div>

      <div style="background: var(--color-bg-subtle); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="font-family: var(--font-display); font-size: 16px; margin-bottom: 12px;">🧠 Explainability Panel</h4>
        <div class="reasoning-text" style="margin: 0; border-left-color: ${severityColor};">
          <strong>Gemma's Reasoning:</strong> ${escapeHtml(r.gemmaResult.reasoning || '(no reasoning returned)')}
          <br><br>
          <strong>Classification:</strong> ${escapeHtml(r.anomalyType)} (VAD verdict: ${escapeHtml(r.vadVerdict)}).
          ${escapeHtml(trackingSummary)}
          Severity computed as VAD × Gemma confidence = ${fmtScore(severity)}.
        </div>
      </div>

      <div class="alert-actions">
        <button class="btn btn-accent" id="btn-confirm-alert" onclick="handleAlertAction('confirm')">
          ✓ Confirm &amp; Dispatch
        </button>
        <button class="btn btn-secondary" id="btn-reject-alert" onclick="handleAlertAction('reject')">
          ✕ Reject
        </button>
        <button class="btn btn-secondary" id="btn-override-alert" onclick="handleAlertAction('override')">
          ✎ Override
        </button>
      </div>
      <p id="alert-action-status" style="font-size: 12px; color: var(--color-text-tertiary); margin-top: 8px;"></p>
    </div>
  `;

  updateStageStatus(7, 'completed');
  completeStageCard(7);
}

// ===== Alert Actions (Human-in-the-loop) =====
function dispatchToPolice() {
  const r = state.pipelineResults;
  const gemma = r.gemmaResult || {};
  const department = DEPARTMENTS.includes(gemma.department_to_notify)
    ? gemma.department_to_notify
    : getDepartmentForAnomaly(gemma.anomaly_type || r.vadVerdict);

  try {
    const bc = new BroadcastChannel('crowdguard_alerts');
    bc.postMessage({
      type: 'ALERT_CONFIRMED',
      severity: r.severity,
      location: CAMERA_LOCATIONS[state.cameraId] || `Unregistered camera (${state.cameraId})`,
      camera: state.cameraId,
      vadScore: r.primaryVadScore,
      gemmaConfidence: r.gemmaConfidence,
      department,
      eventHash: r.logEntry ? r.logEntry.hash : null,
      summary: `${r.anomalyType} (VAD: ${r.vadVerdict}) — ${gemma.reasoning || 'no reasoning'}`
    });
    bc.close();
  } catch (e) {
    console.error('Failed to broadcast police alert:', e);
  }
}

async function handleAlertAction(action) {
  const confirmBtn = document.getElementById('btn-confirm-alert');
  const rejectBtn = document.getElementById('btn-reject-alert');
  const overrideBtn = document.getElementById('btn-override-alert');
  const statusEl = document.getElementById('alert-action-status');

  confirmBtn.disabled = true;
  rejectBtn.disabled = true;
  overrideBtn.disabled = true;

  let entry = null;
  try {
    const data = await postJson('/api/forensic/decision', {
      session_id: state.sessionId,
      event_index: state.pipelineResults.primaryEventIndex,
      action
    });
    if (data.status !== 'success') throw new Error(data.message || 'Could not record decision');
    entry = data.entry;
  } catch (err) {
    statusEl.textContent = `⚠ Decision not recorded: ${err.message}`;
    statusEl.style.color = 'var(--color-danger)';
    confirmBtn.disabled = false;
    rejectBtn.disabled = false;
    overrideBtn.disabled = false;
    return;
  }

  // Only an operator-confirmed alert reaches dispatch — that's the human in the loop.
  if (action === 'confirm') {
    dispatchToPolice();
    confirmBtn.innerHTML = '✓ Alert Confirmed &amp; Dispatched';
    confirmBtn.style.background = 'var(--color-success)';
    confirmBtn.style.color = 'white';
    confirmBtn.style.boxShadow = 'none';
  } else if (action === 'reject') {
    rejectBtn.innerHTML = '✕ Alert Rejected';
    rejectBtn.style.background = 'var(--color-danger)';
    rejectBtn.style.color = 'white';
  } else if (action === 'override') {
    overrideBtn.innerHTML = '✎ Override Applied';
    overrideBtn.style.background = 'var(--color-warning)';
    overrideBtn.style.color = 'white';
  }
  statusEl.textContent = `Decision logged as forensic entry #${entry.seq} (${shortHash(entry.hash)})`;
}

// ===== Comparison Mode (VAD-only vs full pipeline, from this run's real results) =====
function showComparison() {
  const r = state.pipelineResults;
  const events = r.vadEvents || [];
  if (events.length === 0) return;

  const verdictByEvent = {};
  (r.gemmaVerdicts || []).forEach(v => { verdictByEvent[v.event.event_id] = v; });

  const row = (left, right, rightColor, dim) => `
    <div style="padding: 8px 12px; background: white; border-radius: 8px; font-size: 13px; display: flex; justify-content: space-between; gap: 12px;${dim ? ' opacity: 0.4;' : ''}">
      <span>${left}</span>
      <span style="color: ${rightColor}; white-space: nowrap;">${right}</span>
    </div>`;

  document.getElementById('comparison-vad-list').innerHTML = events.map(evt => row(
    `🔴 Alert: ${escapeHtml(evt.predicted_class)} @ ${evt.start_time_sec}s`,
    `Score: ${fmtScore(evt.max_score)}`,
    'var(--color-danger)', false
  )).join('');
  document.getElementById('comparison-vad-summary').textContent =
    `${events.length} alert${events.length > 1 ? 's' : ''} fired at threshold ${r.vadThreshold}`;

  let confirmedCount = 0;
  let unverifiedCount = 0;
  document.getElementById('comparison-full-list').innerHTML = events.map(evt => {
    const v = verdictByEvent[evt.event_id];
    if (!v) {
      unverifiedCount++;
      return row(`⚪ Event ${evt.event_id}: not sent to Gemma (outside top ${MAX_GEMMA_EVENTS})`, 'Unverified', 'var(--color-text-tertiary)', true);
    }
    if (!v.ok) {
      unverifiedCount++;
      return row(`⚠ Event ${evt.event_id}: Gemma error`, 'Needs review', 'var(--color-warning)', false);
    }
    if (v.data.anomaly_confirmed) {
      confirmedCount++;
      return row(`🟢 Confirmed: ${escapeHtml(v.data.gemma_result.anomaly_type || evt.predicted_class)}`, `Severity: ${fmtScore(v.data.severity)}`, 'var(--color-success)', false);
    }
    return row(`⚪ Rejected by Gemma: ${escapeHtml(evt.predicted_class)}`, 'Filtered', 'var(--color-text-tertiary)', true);
  }).join('');

  // Conservative: unverified events still count as alerts reaching the operator.
  const reaching = confirmedCount + unverifiedCount;
  const reduction = Math.round((1 - reaching / events.length) * 100);
  document.getElementById('comparison-full-summary').textContent =
    `${reaching} alert${reaching === 1 ? '' : 's'} reach the operator — ${reduction}% alarm reduction` +
    (unverifiedCount ? ` (${unverifiedCount} unverified counted as alerts)` : '');

  const section = document.getElementById('comparison-section');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ===== Reset Pipeline =====
function resetPipeline() {
  const pipelineSection = document.getElementById('pipeline-processing');
  pipelineSection.classList.remove('visible');

  for (let i = 1; i <= 7; i++) {
    const card = document.getElementById(`stage-${i}`);
    card.classList.remove('visible', 'active', 'completed', 'skipped');
    updateStageStatus(i, 'pending');

    const progress = document.getElementById(`stage-${i}-progress`);
    if (progress) progress.style.width = '0%';

    const results = document.getElementById(`stage-${i}-results`);
    if (results) results.innerHTML = '';
  }

  document.getElementById('comparison-section').style.display = 'none';
  state.pipelineResults = {};
  state.sessionId = null;
  state.pipelineAborted = false;
  state.pipelineFailed = false;
  state.abortReason = '';

  // Without this the button stays disabled on "Analysis Complete" until a page reload.
  const startBtn = document.getElementById('start-analysis-btn');
  startBtn.disabled = false;
  if (startButtonHtml) startBtn.innerHTML = startButtonHtml;
}

// ===== Comparison Toggle =====
function initComparisonToggle() {
  const toggle = document.getElementById('comparison-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('active');
    });
  }
}

// ===== Button Hover Effect =====
function initButtonHoverEffect() {
  document.querySelectorAll('.btn-primary').forEach(btn => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      btn.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
      btn.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
    });
  });
}

// ===== Initialize Everything =====
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initRevealAnimations();
  initPipelineTabs();
  initUpload();
  initPipeline();
  initComparisonToggle();
  initButtonHoverEffect();
});
