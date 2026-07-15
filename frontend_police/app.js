/* ============================================
   CrowdGuard — Police Dispatch Logic
   Simplified: Alert Receiver & Intel Only
   ============================================ */

// Utilities
const sleep = ms => new Promise(r => setTimeout(r, ms));
const genHash = () => Math.random().toString(16).substr(2, 8).toUpperCase();

// Elements
const el = {
  clock: document.getElementById('sys-clock'),
  pipeStatus: document.getElementById('pipeline-status'),
  intelText: document.getElementById('intel-text'),
  forensicLog: document.getElementById('forensic-log'),
  meterFill: document.getElementById('meter-fill'),
  sevVal: document.getElementById('sev-val'),
  body: document.getElementById('body-main'),
  btnAck: document.getElementById('btn-ack'),
  richDataContainer: document.getElementById('rich-data-container'),
  intelLocation: document.getElementById('intel-location'),
  intelCamera: document.getElementById('intel-camera'),
  intelVad: document.getElementById('intel-vad'),
  intelGemma: document.getElementById('intel-gemma'),
  intelDept: document.getElementById('intel-dept')
};

// Clock Update
setInterval(() => {
  const d = new Date();
  el.clock.textContent = d.toISOString().split('T')[1].slice(0, -1);
}, 1000);

// Log output
function addLog(source, msg, type = 'normal') {
  const d = new Date().toISOString().split('T')[1].slice(0, -2);
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  let contentClass = 'log-normal';
  if (type === 'alert') contentClass = 'log-alert';
  if (type === 'hash') contentClass = 'log-hash';
  
  entry.innerHTML = `<span class="log-timestamp">[${d}] ${source}:</span> <span class="${contentClass}">${msg}</span>`;
  el.forensicLog.appendChild(entry);
  el.forensicLog.scrollTop = el.forensicLog.scrollHeight;
}

// Typing effect for Intel
async function typeIntel(text) {
  el.intelText.textContent = '';
  for (let i = 0; i < text.length; i++) {
    el.intelText.textContent += text[i];
    await sleep(20);
  }
}

// Ack Button
el.btnAck.addEventListener('click', () => {
  el.body.classList.remove('global-alert');
  el.btnAck.disabled = true;
  el.btnAck.textContent = "Threat Acknowledged";
  el.btnAck.className = "btn"; // reset style
  el.btnAck.style.border = "1px solid var(--color-border)";
  el.btnAck.style.color = "var(--color-text-tertiary)";
  el.pipeStatus.className = 'status-indicator ok';
  el.pipeStatus.textContent = 'LINK: SECURE';
  addLog('OPR', 'Threat acknowledged by Dispatch', 'hash');
});

// ===== Broadcast Channel for Cross-Dashboard Alerts =====
try {
  const bc = new BroadcastChannel('crowdguard_alerts');
  bc.onmessage = (event) => {
    if (event.data.type === 'ALERT_CONFIRMED') {
      
      const payload = event.data;
      
      // Update Severity Meter
      const sev = payload.severity ? payload.severity.toFixed(2) : '0.00';
      el.sevVal.textContent = sev;
      el.meterFill.style.width = `${sev * 100}%`;

      // Trigger global alert
      el.body.classList.add('global-alert');
      el.pipeStatus.className = 'status-indicator alert';
      el.pipeStatus.textContent = 'INCOMING DISPATCH';
      
      // Populate Rich Data
      el.richDataContainer.style.display = 'grid';
      el.intelLocation.textContent = payload.location || 'UNKNOWN';
      el.intelCamera.textContent = payload.camera || 'UNKNOWN';
      el.intelVad.textContent = payload.vadScore ? payload.vadScore.toFixed(2) : '--';
      el.intelGemma.textContent = payload.gemmaConfidence ? payload.gemmaConfidence.toFixed(2) : '--';
      el.intelDept.textContent = payload.department || 'UNKNOWN';

      // Logs & Intel
      addLog('DISPATCH', `[URGENT] INCOMING ALERT FROM HQ: ${payload.summary}`, 'alert');
      typeIntel(`DISPATCH INTEL: External alert confirmed by HQ. Subject flagged for immediate interception in ${payload.location || 'Sector 7'}. ${payload.summary}`);
      
      // Enable Ack Button
      el.btnAck.disabled = false;
      el.btnAck.textContent = "Acknowledge Alert";
      el.btnAck.className = "btn btn-danger";
      el.btnAck.style.border = "none";
      el.btnAck.style.color = "white";
    }
  };
} catch (e) {
  console.error('BroadcastChannel error', e);
}
