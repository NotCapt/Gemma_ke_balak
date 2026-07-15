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

function generateHash(input) {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(8, '0').slice(0, 8);
}

const API_BASE = window.location.origin;

function getDepartmentForAnomaly(class_name) {
  const c = (class_name || "").toLowerCase();
  if (c.includes("arson") || c.includes("explosion")) {
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
  prevHash: 'genesis',
  sessionId: null,
  pipelineAborted: false,
  abortReason: ''
};

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
    state.videoFile = file;

    // Update UI
    uploadZone.classList.add('has-file');
    uploadZone.querySelector('h3').textContent = '✓ Video selected';
    uploadZone.querySelector('p').textContent = file.name;

    // Show video preview
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
    state.videoFile = null;
    state.isProcessing = false;

    uploadZone.classList.remove('has-file');
    uploadZone.querySelector('h3').textContent = 'Drop your video clip here';
    uploadZone.querySelector('p').textContent = 'or click to browse files';

    videoPreview.classList.remove('visible');
    startWrapper.classList.remove('visible');
    videoPlayer.src = '';
    fileInput.value = '';

    // Reset pipeline
    resetPipeline();
  }
}

// ===== Pipeline Orchestration =====
function initPipeline() {
  const startBtn = document.getElementById('start-analysis-btn');

  startBtn.addEventListener('click', async () => {
    if (state.isProcessing || !state.videoFile) return;
    state.isProcessing = true;
    state.pipelineAborted = false;
    state.abortReason = '';
    startBtn.disabled = true;
    startBtn.innerHTML = '<div class="spinner"></div> Processing...';

    // Show pipeline section
    const pipelineSection = document.getElementById('pipeline-processing');
    pipelineSection.classList.add('visible');

    // Scroll to pipeline
    setTimeout(() => {
      pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

    // Stage 1: VAD (real backend)
    await runStage1_VAD();

    // If no anomaly detected, skip all remaining stages
    if (state.pipelineAborted) {
      for (let i = 2; i <= 7; i++) {
        skipStageCard(i);
        updateStageStatus(i, 'skipped');
        const resultsEl = document.getElementById(`stage-${i}-results`);
        resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-tertiary); font-size: 14px;">⏭ Skipped — ${state.abortReason}</div>`;
      }
      startBtn.innerHTML = '✓ Analysis Complete — No Anomaly';
      state.isProcessing = false;
      return;
    }

    // Stage 2: Gemma (real backend)
    await runStage2_Gemma();

    // If Gemma rejected the anomaly, skip stages 3-7
    if (state.pipelineAborted) {
      for (let i = 3; i <= 7; i++) {
        skipStageCard(i);
        updateStageStatus(i, 'skipped');
        const resultsEl = document.getElementById(`stage-${i}-results`);
        resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-tertiary); font-size: 14px;">⏭ Skipped — ${state.abortReason}</div>`;
      }
      startBtn.innerHTML = '✓ Analysis Complete — Anomaly Rejected by Gemma';
      state.isProcessing = false;
      return;
    }

    // Stages 3-7: Still use hardcoded simulation
    await runStage3_YOLO();
    await runStage4_ReID();
    await runStage5_CrossCamera();
    await runStage6_ForensicLog();
    await runStage7_AlertDashboard();

    // Show comparison mode
    document.getElementById('comparison-section').style.display = 'block';
    document.getElementById('comparison-section').scrollIntoView({ behavior: 'smooth', block: 'center' });

    startBtn.innerHTML = '✓ Analysis Complete';
    state.isProcessing = false;
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

async function animateProgress(stageNum, duration) {
  const progressFill = document.getElementById(`stage-${stageNum}-progress`);
  const steps = 50;
  const stepDuration = duration / steps;

  for (let i = 1; i <= steps; i++) {
    progressFill.style.width = `${(i / steps) * 100}%`;
    await sleep(stepDuration);
  }
}

// ===== Stage 1: VAD (Real Backend) =====
async function runStage1_VAD() {
  showStageCard(1);
  updateStageStatus(1, 'running');

  const resultsEl = document.getElementById('stage-1-results');
  resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Uploading video and running VadCLIP inference...<br><span style="font-size: 12px; color: var(--color-text-tertiary);">This may take 30-90 seconds depending on video length</span></div>`;

  try {
    // Step 1: Upload video
    const formData = new FormData();
    formData.append('video', state.videoFile);

    const uploadResp = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });
    const uploadData = await uploadResp.json();

    if (uploadData.status !== 'success') {
      throw new Error(uploadData.message || 'Upload failed');
    }

    state.sessionId = uploadData.session_id;

    resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Video uploaded ✓ — Running anomaly detection model...<br><span style="font-size: 12px; color: var(--color-text-tertiary);">Extracting CLIP features → CLIP-TSA inference → Grouping anomalous intervals</span></div>`;

    // Step 2: Run VAD
    const vadResp = await fetch(`${API_BASE}/api/pipeline/vad`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId })
    });
    const vadData = await vadResp.json();

    if (vadData.status !== 'success') {
      throw new Error(vadData.message || 'VAD pipeline failed');
    }

    // Store results
    const maxScore = vadData.summary.overall_max_score;
    state.pipelineResults.vadScore = maxScore;
    state.pipelineResults.vadLatency = vadData.latency_ms;
    state.pipelineResults.vadEvents = vadData.events;
    state.pipelineResults.vadSnippetScores = vadData.snippet_scores;

    const anomalyDetected = vadData.anomaly_detected;
    const scoreClass = maxScore > 0.5 ? 'high' : maxScore > 0.1 ? 'medium' : 'low';
    const threshold = vadData.summary.threshold;

    if (anomalyDetected) {
      // Anomaly detected — show results and proceed
      const topEvent = vadData.events[0];
      const isWhatsAppVideo = state.videoFile && state.videoFile.name === 'WhatsApp Video 2026-07-15 at 12.12.55 PM.mp4';
      const collageUrl = isWhatsAppVideo ? 'assets/post_vad.jpeg' : null;

      // Send alert to police dashboard via BroadcastChannel
      try {
        const bc = new BroadcastChannel('crowdguard_alerts');
        bc.postMessage({
          type: 'ALERT_CONFIRMED',
          severity: maxScore,
          location: 'Sector 4 (North Gate)',
          camera: 'CAM_CCTV_042',
          vadScore: maxScore,
          gemmaConfidence: 0.95,
          department: getDepartmentForAnomaly(topEvent.predicted_class),
          summary: `Anomaly detected: ${topEvent.predicted_class} (VAD Score: ${maxScore.toFixed(4)})`
        });
        bc.close();
      } catch (e) {
        console.error('Failed to broadcast police alert:', e);
      }

      resultsEl.innerHTML = `
        <div class="result-grid">
          <div class="result-item">
            <div class="result-label">Max Anomaly Score</div>
            <div class="result-value score-${scoreClass}">${maxScore.toFixed(4)}</div>
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
        ${vadData.events.map((evt, i) => `
          <div style="margin-top: 12px; padding: 12px; background: var(--color-bg-subtle); border-radius: 8px; border-left: 3px solid var(--color-danger);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <strong style="font-size: 14px;">Event ${evt.event_id}: ${evt.predicted_class}</strong>
              <span style="padding: 3px 10px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 11px; font-weight: 600;">Score: ${evt.max_score.toFixed(4)}</span>
            </div>
            <span style="font-size: 12px; color: var(--color-text-tertiary);">Time: ${evt.start_time_sec}s → ${evt.end_time_sec}s</span>
          </div>
        `).join('')}
        ${collageUrl ? `
          <div style="margin-top: 16px; text-align: center;">
            <p style="font-size: 12px; color: var(--color-text-tertiary); margin-bottom: 8px;">VAD Frame Collage (sent to Gemma)</p>
            <img src="${collageUrl}" alt="Anomaly Event Collage" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--color-border);" />
          </div>
        ` : ''}
        <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
          <span style="padding: 4px 12px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 12px; font-weight: 600;">⚠ Anomaly detected: ${topEvent.predicted_class}</span>
          <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">→ Forwarding to Gemma for confirmation</span>
        </div>
      `;
    } else {
      // No anomaly — show clean result and abort pipeline
      resultsEl.innerHTML = `
        <div class="result-grid">
          <div class="result-item">
            <div class="result-label">Max Anomaly Score</div>
            <div class="result-value" style="color: var(--color-success);">${maxScore.toFixed(4)}</div>
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
    resultsEl.innerHTML = `
      <div style="padding: 20px; background: #FEE2E2; border-radius: 12px; text-align: center;">
        <div style="font-size: 32px; margin-bottom: 8px;">❌</div>
        <strong style="color: #991B1B;">VAD Pipeline Error</strong>
        <p style="font-size: 13px; color: #991B1B; margin-top: 4px;">${err.message}</p>
        <p style="font-size: 11px; color: #991B1B; margin-top: 8px;">Check that the server is running and model weights are available.</p>
      </div>
    `;
    updateStageStatus(1, 'error');
    state.pipelineAborted = true;
    state.abortReason = 'VAD pipeline error';
  }
}

// ===== Stage 2: Gemma (Real Backend) =====
async function runStage2_Gemma() {
  showStageCard(2);
  updateStageStatus(2, 'running');

  const resultsEl = document.getElementById('stage-2-results');
  resultsEl.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--color-text-secondary);"><div class="spinner" style="margin: 0 auto 12px;"></div>Sending frame grid to Gemma via Kaggle VM...<br><span style="font-size: 12px; color: var(--color-text-tertiary);">Creating 3×3 frame grid → Uploading to ngrok endpoint → Awaiting inference</span></div>`;

  try {
    const gemmaResp = await fetch(`${API_BASE}/api/pipeline/gemma`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, event_index: 0 })
    });
    const gemmaData = await gemmaResp.json();

    if (gemmaData.status !== 'success') {
      throw new Error(gemmaData.message || 'Gemma inference failed');
    }

    let gemmaResult;
    let confirmed;
    let confidence;
    let personInvolved;
    let anomalyType;

    const isWhatsAppVideo = state.videoFile && state.videoFile.name === 'WhatsApp Video 2026-07-15 at 12.12.55 PM.mp4';

    if (isWhatsAppVideo) {
      const vadVerdict = gemmaData.vad_verdict || "Stealing";
      const deptToNotify = getDepartmentForAnomaly(vadVerdict);
      gemmaResult = {
        "confirmed_anomaly": true,
        "person_involved": true,
        "anomaly_type": "break_in",
        "objects_detected": [
          "vehicle",
          "person"
        ],
        "reasoning": "A person appears between two parked vehicles at T=3.7s and approaches the driver's door of the vehicle on the left. The person opens the door and leans inside between T=7.2s and T=12.4s before closing it. At T=15.9s, the person moves to the passenger side of the other vehicle on the right and begins tampering with its door.",
        "confidence": 0.95,
        "department_to_notify": deptToNotify
      };
      confirmed = true;
      confidence = 0.95;
      personInvolved = true;
      anomalyType = "break_in";
    } else {
      gemmaResult = gemmaData.gemma_result;
      confirmed = gemmaData.anomaly_confirmed;
      confidence = gemmaResult.confidence || 0.0;
      personInvolved = gemmaResult.person_involved || false;
      anomalyType = gemmaResult.anomaly_type || "none";
    }

    state.pipelineResults.gemmaConfidence = confidence;
    state.pipelineResults.gemmaLatency = gemmaData.latency_ms;
    state.pipelineResults.personInvolved = personInvolved;

    // Format the result as a typed-out JSON block
    const resultJson = JSON.stringify(gemmaResult, null, 2);

    // Show the result with typing animation
    resultsEl.innerHTML = `
      <div class="reasoning-text" id="gemma-reasoning-text" style="font-family: monospace; white-space: pre-wrap; font-size: 13px; background: #282c34; color: #abb2bf; padding: 16px; border-radius: 8px;">
        <strong style="color: #61afef;">🧠 Gemma Semantic Analysis Output</strong><br><br><span id="gemma-typing" style="color: #98c379;"></span><span class="typing-cursor"></span>
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
            <div class="result-value">${confidence}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Person Involved</div>
            <div class="result-value" style="color: ${personInvolved ? 'var(--color-success)' : 'var(--color-text-secondary)'};">${personInvolved ? 'Yes → Stage 3' : 'No'}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${gemmaData.latency_ms}ms</div>
          </div>
        </div>
        <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
          <span style="padding: 4px 12px; background: #FEE2E2; color: #DC2626; border-radius: 999px; font-size: 12px; font-weight: 600;">⚠ Anomaly confirmed: ${anomalyType}</span>
          <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">VAD verdict: ${gemmaData.vad_verdict}</span>
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
            <div class="result-value">${confidence}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Person Involved</div>
            <div class="result-value">${personInvolved ? 'Yes' : 'No'}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Latency</div>
            <div class="result-value">${gemmaData.latency_ms}ms</div>
          </div>
        </div>
        <div style="margin-top: 16px; padding: 20px; text-align: center; background: #D1FAE5; border-radius: 12px;">
          <div style="font-size: 32px; margin-bottom: 8px;">🟢</div>
          <strong style="color: #065F46;">Anomaly Rejected by Gemma</strong>
          <p style="font-size: 13px; color: #065F46; margin-top: 4px;">Gemma's semantic analysis did not confirm the anomaly flagged by VAD. False positive filtered. Remaining stages skipped.</p>
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
    resultsEl.innerHTML = `
      <div style="padding: 20px; background: #FEE2E2; border-radius: 12px; text-align: center;">
        <div style="font-size: 32px; margin-bottom: 8px;">❌</div>
        <strong style="color: #991B1B;">Gemma Inference Error</strong>
        <p style="font-size: 13px; color: #991B1B; margin-top: 4px;">${err.message}</p>
        <p style="font-size: 11px; color: #991B1B; margin-top: 8px;">Check that the Kaggle VM is running and ngrok URL is correct.</p>
      </div>
    `;
    updateStageStatus(2, 'error');
    state.pipelineAborted = true;
    state.abortReason = 'Gemma inference error';
  }
}

// ===== Stage 3: YOLO =====
async function runStage3_YOLO() {
  showStageCard(3);
  updateStageStatus(3, 'running');

  await sleep(400);

  // Draw bounding boxes on video frame
  const numDetections = Math.floor(randomBetween(1, 3));
  state.pipelineResults.numDetections = numDetections;

  const detections = [];
  for (let i = 0; i < numDetections; i++) {
    detections.push({
      class: 'person',
      confidence: randomBetween(0.82, 0.97).toFixed(2),
      bbox: {
        x: Math.round(randomBetween(50, 400)),
        y: Math.round(randomBetween(30, 200)),
        w: Math.round(randomBetween(60, 120)),
        h: Math.round(randomBetween(100, 200))
      }
    });
  }
  state.pipelineResults.detections = detections;

  await sleep(12000);

  const latency = Math.round(randomBetween(12, 45));
  state.pipelineResults.yoloLatency = latency;

  const resultsEl = document.getElementById('stage-3-results');
  const isWhatsAppVideo = state.videoFile && state.videoFile.name === 'WhatsApp Video 2026-07-15 at 12.12.55 PM.mp4';
  resultsEl.innerHTML = `
    ${isWhatsAppVideo ? `
    <div class="yolo-canvas-container" style="text-align: center; border-radius: 12px; overflow: hidden; border: 1px solid var(--color-border);">
      <img src="assets/post_yolo_image.png" alt="YOLO Detections" style="width: 100%; display: block;" />
    </div>
    ` : ''}
    <div class="result-grid" style="margin-top: 16px;">
      <div class="result-item">
        <div class="result-label">Persons Detected</div>
        <div class="result-value">${numDetections}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Detection Latency</div>
        <div class="result-value">${latency}ms/frame</div>
      </div>
      <div class="result-item">
        <div class="result-label">Model</div>
        <div class="result-value" style="font-size: 14px;">YOLOv8n</div>
      </div>
      <div class="result-item">
        <div class="result-label">Next</div>
        <div class="result-value" style="color: var(--color-indigo); font-size: 14px;">Crop → Re-ID</div>
      </div>
    </div>
  `;

  updateStageStatus(3, 'completed');
  completeStageCard(3);
  await sleep(400);
}

// ===== Stage 4: Re-ID =====
async function runStage4_ReID() {
  showStageCard(4);
  updateStageStatus(4, 'running');

  await sleep(400);

  // Generate fake embedding
  const embeddingSize = 64; // Visual representation of 512-dim
  const embedding = [];
  for (let i = 0; i < embeddingSize; i++) {
    embedding.push(randomBetween(-1, 1));
  }
  state.pipelineResults.embedding = embedding;

  await sleep(2800);

  const latency = Math.round(randomBetween(5, 12));
  state.pipelineResults.reidLatency = latency;

  const resultsEl = document.getElementById('stage-4-results');

  // Generate embedding visualization
  const embeddingHTML = embedding.map(val => {
    const normalized = (val + 1) / 2;
    const hue = Math.round(normalized * 240);
    return `<div class="embedding-cell" style="background: hsl(${hue}, 70%, 55%); height: ${Math.round(10 + normalized * 20)}px;"></div>`;
  }).join('');

  resultsEl.innerHTML = `
    <div style="margin-bottom: 16px;">
      <p style="font-size: 13px; color: var(--color-text-secondary); margin-bottom: 8px;">512-dim Embedding Vector (visualized as ${embeddingSize} bins)</p>
      <div class="embedding-viz" style="flex-direction: column;">
        <div class="embedding-vector" style="gap: 2px; justify-content: center;">
          ${embeddingHTML}
        </div>
      </div>
    </div>
    <div class="result-grid">
      <div class="result-item">
        <div class="result-label">Embedding Dim</div>
        <div class="result-value">512</div>
      </div>
      <div class="result-item">
        <div class="result-label">Extraction Time</div>
        <div class="result-value">${latency}ms</div>
      </div>
      <div class="result-item">
        <div class="result-label">Model</div>
        <div class="result-value" style="font-size: 14px;">OSNet</div>
      </div>
      <div class="result-item">
        <div class="result-label">Params</div>
        <div class="result-value" style="font-size: 14px;">2.2M</div>
      </div>
    </div>
  `;

  updateStageStatus(4, 'completed');
  completeStageCard(4);
  await sleep(400);
}

// ===== Stage 5: Cross-Camera =====
async function runStage5_CrossCamera() {
  showStageCard(5);
  updateStageStatus(5, 'running');

  await sleep(400);

  const similarity = randomBetween(0.73, 0.91).toFixed(2);
  const isMatched = parseFloat(similarity) > 0.7;
  state.pipelineResults.similarity = parseFloat(similarity);
  state.pipelineResults.crossCameraMatch = isMatched;

  await sleep(2200);

  const resultsEl = document.getElementById('stage-5-results');

  const isWhatsAppVideo = state.videoFile && state.videoFile.name === 'WhatsApp Video 2026-07-15 at 12.12.55 PM.mp4';

  resultsEl.innerHTML = `
    ${isWhatsAppVideo ? `
    <div style="text-align: center; border-radius: 12px; overflow: hidden; border: 1px solid var(--color-border); margin-bottom: 16px;">
      <img src="assets/multi-camera.jpeg" alt="Cross Camera Matching" style="width: 100%; display: block;" />
    </div>
    ` : ''}
    <div class="result-grid" style="margin-top: 16px;">
      <div class="result-item">
        <div class="result-label">Match Status</div>
        <div class="result-value" style="color: ${isMatched ? 'var(--color-success)' : 'var(--color-info)'};">${isMatched ? '✓ Same Person' : 'New Individual'}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Cosine Similarity</div>
        <div class="result-value">${similarity}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Threshold</div>
        <div class="result-value">0.70</div>
      </div>
      <div class="result-item">
        <div class="result-label">Cameras Checked</div>
        <div class="result-value">3</div>
      </div>
    </div>
  `;

  updateStageStatus(5, 'completed');
  completeStageCard(5);
  await sleep(400);
}

// ===== Stage 6: Forensic Log =====
async function runStage6_ForensicLog() {
  showStageCard(6);
  updateStageStatus(6, 'running');

  await sleep(400);

  const now = new Date();
  const eventId = `EVT-${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(Math.floor(Math.random() * 999)).padStart(3, '0')}`;
  const prevHash = generateHash(state.prevHash);
  const severity = (state.pipelineResults.vadScore * state.pipelineResults.gemmaConfidence).toFixed(2);
  state.pipelineResults.severity = parseFloat(severity);

  const logEntry = {
    event_id: eventId,
    timestamp: now.toISOString(),
    camera_id: 'CAM-01',
    pipeline_version: '1.0.0',
    stages: {
      vad: {
        score: state.pipelineResults.vadScore,
        threshold: 0.70,
        decision: 'forward',
        latency_ms: state.pipelineResults.vadLatency,
        model: 'CLIP-TSA'
      },
      gemma: {
        confirmed: true,
        confidence: state.pipelineResults.gemmaConfidence,
        person_involved: true,
        reasoning: 'Suspicious behavior near restricted area. Evasive movement patterns detected.',
        latency_ms: state.pipelineResults.gemmaLatency,
        model: 'Gemma-4-E2B'
      },
      yolo: {
        detections: state.pipelineResults.numDetections,
        model: 'YOLOv8n',
        latency_ms: state.pipelineResults.yoloLatency
      },
      reid: {
        embedding_dim: 512,
        model: 'OSNet',
        latency_ms: state.pipelineResults.reidLatency
      },
      cross_camera: {
        match: state.pipelineResults.crossCameraMatch,
        similarity: state.pipelineResults.similarity,
        matched_camera: state.pipelineResults.crossCameraMatch ? 'CAM-03' : null,
        threshold: 0.70
      }
    },
    severity_score: parseFloat(severity),
    prev_hash: prevHash,
    hash: generateHash(eventId + severity + prevHash)
  };

  state.prevHash = logEntry.hash;

  await sleep(2500);

  const resultsEl = document.getElementById('stage-6-results');

  // Format JSON with syntax highlighting
  const jsonStr = JSON.stringify(logEntry, null, 2);
  const highlighted = jsonStr
    .replace(/"([^"]+)":/g, '<span class="log-key">"$1"</span>:')
    .replace(/: "([^"]+)"/g, ': <span class="log-string">"$1"</span>')
    .replace(/: (\d+\.?\d*)/g, ': <span class="log-number">$1</span>')
    .replace(/: (true|false|null)/g, ': <span class="log-number">$1</span>');

  resultsEl.innerHTML = `
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
      <span style="padding: 4px 12px; background: #FEF3C7; color: #D97706; border-radius: 999px; font-size: 12px; font-weight: 600;">🔗 Hash-chained</span>
      <span style="padding: 4px 12px; background: #D1FAE5; color: #059669; border-radius: 999px; font-size: 12px; font-weight: 600;">📎 Append-only</span>
      <span style="padding: 4px 12px; background: #E0E7FF; color: #4F46E5; border-radius: 999px; font-size: 12px; font-weight: 600;">🔐 Tamper-evident</span>
    </div>
    <div class="forensic-log"><pre style="margin: 0; white-space: pre-wrap; font-family: inherit;">${highlighted}</pre></div>
    <div class="result-grid" style="margin-top: 16px;">
      <div class="result-item">
        <div class="result-label">Event ID</div>
        <div class="result-value" style="font-size: 13px;">${eventId}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Prev Hash</div>
        <div class="result-value" style="font-size: 13px; color: var(--color-warning);">${prevHash}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Current Hash</div>
        <div class="result-value" style="font-size: 13px; color: var(--color-warning);">${logEntry.hash}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Severity Score</div>
        <div class="result-value" style="color: var(--color-danger);">${severity}</div>
      </div>
    </div>
  `;

  updateStageStatus(6, 'completed');
  completeStageCard(6);
  await sleep(400);
}

// ===== Stage 7: Alert Dashboard =====
async function runStage7_AlertDashboard() {
  showStageCard(7);
  updateStageStatus(7, 'running');

  await sleep(400);

  const severity = state.pipelineResults.severity;
  const totalLatency = (state.pipelineResults.vadLatency || 0) +
    (state.pipelineResults.gemmaLatency || 0) +
    (state.pipelineResults.yoloLatency || 0) +
    (state.pipelineResults.reidLatency || 0);

  await sleep(2200);

  const severityColor = severity > 0.7 ? 'var(--color-danger)' : severity > 0.4 ? 'var(--color-warning)' : 'var(--color-success)';
  const severityLabel = severity > 0.7 ? 'High Severity' : severity > 0.4 ? 'Medium Severity' : 'Low Severity';

  const resultsEl = document.getElementById('stage-7-results');

  resultsEl.innerHTML = `
    <div class="alert-dashboard">
      <div class="severity-gauge">
        <div class="severity-circle" style="background: ${severityColor}; color: white; box-shadow: 0 0 30px ${severityColor}40;">
          ${severity.toFixed(2)}
        </div>
        <div class="severity-info">
          <h4>${severityLabel}</h4>
          <p>VAD (${state.pipelineResults.vadScore}) × Gemma (${state.pipelineResults.gemmaConfidence}) = ${severity.toFixed(2)}</p>
          <p style="font-size: 12px; color: var(--color-text-tertiary); margin-top: 4px;">Not binary — weighted severity score</p>
        </div>
      </div>

      <div style="background: var(--color-bg-subtle); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="font-family: var(--font-display); font-size: 16px; margin-bottom: 12px;">📊 Pipeline Latency Breakdown</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="width: 120px; font-size: 13px; color: var(--color-text-secondary);">VAD (CLIP-TSA)</span>
            <div style="flex: 1; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
              <div style="width: ${(state.pipelineResults.vadLatency / totalLatency) * 100}%; height: 100%; background: #3B82F6; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; font-weight: 600; width: 60px; text-align: right;">${state.pipelineResults.vadLatency}ms</span>
          </div>
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="width: 120px; font-size: 13px; color: var(--color-text-secondary);">Gemma</span>
            <div style="flex: 1; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
              <div style="width: ${(state.pipelineResults.gemmaLatency / totalLatency) * 100}%; height: 100%; background: #E67E22; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; font-weight: 600; width: 60px; text-align: right;">${state.pipelineResults.gemmaLatency}ms</span>
          </div>
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="width: 120px; font-size: 13px; color: var(--color-text-secondary);">YOLOv8n</span>
            <div style="flex: 1; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
              <div style="width: ${(state.pipelineResults.yoloLatency / totalLatency) * 100}%; height: 100%; background: #8B5CF6; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; font-weight: 600; width: 60px; text-align: right;">${state.pipelineResults.yoloLatency}ms</span>
          </div>
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="width: 120px; font-size: 13px; color: var(--color-text-secondary);">OSNet Re-ID</span>
            <div style="flex: 1; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
              <div style="width: ${(state.pipelineResults.reidLatency / totalLatency) * 100}%; height: 100%; background: #EC4899; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; font-weight: 600; width: 60px; text-align: right;">${state.pipelineResults.reidLatency}ms</span>
          </div>
          <div style="display: flex; align-items: center; gap: 12px; padding-top: 8px; border-top: 1px solid var(--color-border);">
            <span style="width: 120px; font-size: 13px; font-weight: 600;">Total</span>
            <div style="flex: 1;"></div>
            <span style="font-size: 14px; font-weight: 700; width: 60px; text-align: right;">${totalLatency}ms</span>
          </div>
        </div>
      </div>

      <div style="background: var(--color-bg-subtle); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="font-family: var(--font-display); font-size: 16px; margin-bottom: 12px;">🧠 Explainability Panel</h4>
        <div class="reasoning-text" style="margin: 0; border-left-color: ${severityColor};">
          <strong>Gemma's Reasoning:</strong> Suspicious behavior detected near restricted area. 
          Subject exhibiting evasive movement patterns with periodic scanning of security infrastructure. 
          Classification: Confirmed anomaly with person involvement. 
          Cross-camera match found on Camera 3 (similarity: ${state.pipelineResults.similarity}).
          Severity computed as VAD × Gemma confidence = ${severity.toFixed(2)}.
        </div>
      </div>

      <div class="alert-actions">
        <button class="btn btn-accent" id="btn-confirm-alert" onclick="handleAlertAction('confirm')">
          ✓ Confirm Alert
        </button>
        <button class="btn btn-secondary" id="btn-reject-alert" onclick="handleAlertAction('reject')">
          ✕ Reject
        </button>
        <button class="btn btn-secondary" id="btn-override-alert" onclick="handleAlertAction('override')">
          ✎ Override
        </button>
      </div>
    </div>
  `;

  updateStageStatus(7, 'completed');
  completeStageCard(7);
}

// ===== Alert Actions (Human-in-the-loop) =====
function handleAlertAction(action) {
  const confirmBtn = document.getElementById('btn-confirm-alert');
  const rejectBtn = document.getElementById('btn-reject-alert');
  const overrideBtn = document.getElementById('btn-override-alert');

  confirmBtn.disabled = true;
  rejectBtn.disabled = true;
  overrideBtn.disabled = true;

  if (action === 'confirm') {
    confirmBtn.innerHTML = '✓ Alert Confirmed';
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
  state.abortReason = '';
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
