(() => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const pickFilesBtn = document.getElementById('pickFilesBtn');
  const loadSamplesBtn = document.getElementById('loadSamplesBtn');
  const statusBanner = document.getElementById('statusBanner');
  const resultsTbody = document.getElementById('resultsTbody');
  const exportJsonBtn = document.getElementById('exportJsonBtn');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const clearResultsBtn = document.getElementById('clearResultsBtn');

  const statTotal = document.getElementById('statTotal');
  const statConfirmed = document.getElementById('statConfirmed');
  const statCarrier = document.getElementById('statCarrier');
  const statClean = document.getElementById('statClean');

  const inspectDialog = document.getElementById('inspectDialog');
  const inspectDialogTitle = document.getElementById('inspectDialogTitle');
  const inspectDialogSubtitle = document.getElementById('inspectDialogSubtitle');
  const tablist = inspectDialog.querySelector('[role="tablist"], .tablist');
  const tabs = Array.from(inspectDialog.querySelectorAll('[role="tab"]'));
  const panels = Array.from(inspectDialog.querySelectorAll('[role="tabpanel"]'));

  const panelOverviewContent = document.getElementById('panelOverviewContent');
  const panelC2paContent = document.getElementById('panelC2paContent');
  const panelPixelContent = document.getElementById('panelPixelContent');
  const panelContainerContent = document.getElementById('panelContainerContent');
  const panelDspContent = document.getElementById('panelDspContent');

  let currentResults = [];

  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function showStatus(msg) {
    statusBanner.textContent = msg;
    statusBanner.hidden = false;
  }

  function hideStatus() {
    statusBanner.hidden = true;
    statusBanner.textContent = '';
  }

  function verdictBadge(verdict) {
    if (verdict === 'WATERMARK_CONFIRMED') {
      return '<span class="badge badge-confirmed">Confirmed</span>';
    }
    if (verdict === 'WATERMARK_CARRIER_DETECTED') {
      return '<span class="badge badge-carrier">Signal Detected</span>';
    }
    if (verdict === 'NO_WATERMARK_FOUND') {
      return '<span class="badge badge-clean">None Found</span>';
    }
    return `<span class="badge badge-error">${escapeHtml(verdict)}</span>`;
  }

  function updateSummary() {
    const total = currentResults.length;
    const confirmed = currentResults.filter((r) => r.verdict === 'WATERMARK_CONFIRMED').length;
    const carrier = currentResults.filter((r) => r.verdict === 'WATERMARK_CARRIER_DETECTED').length;
    const clean = currentResults.filter((r) => r.verdict === 'NO_WATERMARK_FOUND').length;

    statTotal.textContent = String(total);
    statConfirmed.textContent = String(confirmed);
    statCarrier.textContent = String(carrier);
    statClean.textContent = String(clean);

    const hasData = total > 0;
    exportJsonBtn.disabled = !hasData;
    exportCsvBtn.disabled = !hasData;
    clearResultsBtn.disabled = !hasData;
  }

  function renderTable() {
    if (currentResults.length === 0) {
      resultsTbody.innerHTML =
        '<tr class="empty-row"><td colspan="6">No images scanned. Upload files or click <strong>Load Demo Samples</strong>.</td></tr>';
      updateSummary();
      return;
    }

    resultsTbody.innerHTML = currentResults
      .map((r, idx) => {
        const layers = (r.triggered_layers && r.triggered_layers.length)
          ? r.triggered_layers.join(', ')
          : '-';
        const primaryEvidence = (r.all_evidence && r.all_evidence.length)
          ? r.all_evidence[0]
          : (r.error || 'No watermark or carrier signal found');
        const platform = r.platform || '-';

        return `
          <tr>
            <th scope="row" title="${escapeHtml(r.filename)}">${escapeHtml(r.filename)}</th>
            <td>${verdictBadge(r.verdict)}</td>
            <td class="single-line" title="${escapeHtml(platform)}">${escapeHtml(platform)}</td>
            <td class="single-line" title="${escapeHtml(layers)}">${escapeHtml(layers)}</td>
            <td class="single-line" title="${escapeHtml(primaryEvidence)}">${escapeHtml(primaryEvidence)}</td>
            <td><button type="button" class="btn btn-secondary btn-sm inspect-btn" data-index="${idx}">Details</button></td>
          </tr>
        `;
      })
      .join('');

    updateSummary();
  }

  async function analyzeFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;

    showStatus(`Scanning ${files.length} file(s)...`);
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok) {
        showStatus(`Scan failed: ${data.error || resp.statusText}`);
        return;
      }
      currentResults = currentResults.concat(data.results || []);
      renderTable();
      showStatus(`Scanned ${data.results.length} file(s). Total: ${currentResults.length}.`);
    } catch (err) {
      showStatus(`Network error: ${err.message}`);
    }
  }

  async function loadDemoSamples() {
    showStatus('Loading demo samples...');
    loadSamplesBtn.disabled = true;
    try {
      const resp = await fetch('/api/load-samples', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) {
        showStatus(`Failed to load samples: ${data.error || resp.statusText}`);
        return;
      }
      currentResults = data.results || [];
      renderTable();
      showStatus(`Loaded ${currentResults.length} demo samples.`);
    } catch (err) {
      showStatus(`Failed to load samples: ${err.message}`);
    } finally {
      loadSamplesBtn.disabled = false;
    }
  }

  // Tab selection following modern-web-guidance (tabs)
  function selectTab(selectedTab, focusTab = false) {
    for (const tab of tabs) {
      const isSelected = tab === selectedTab;
      tab.setAttribute('aria-selected', String(isSelected));
      tab.toggleAttribute('focusgroupstart', isSelected);
      if (!Object.hasOwn(HTMLElement.prototype, 'focusGroup')) {
        tab.tabIndex = isSelected ? 0 : -1;
      }
    }

    for (const panel of panels) {
      const isMatch = panel.getAttribute('aria-labelledby') === selectedTab.id;
      if (isMatch) {
        panel.removeAttribute('hidden');
        panel.tabIndex = 0;
      } else {
        panel.setAttribute('hidden', 'until-found');
        panel.removeAttribute('tabindex');
      }
    }

    selectedTab.scrollIntoView({ block: 'nearest', inline: 'nearest', container: 'nearest' });
    if (focusTab) {
      selectedTab.focus();
    }
  }

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => selectTab(tab));
  });

  panels.forEach((panel) => {
    panel.addEventListener('beforematch', () => {
      const targetTab = tabs.find((t) => t.id === panel.getAttribute('aria-labelledby'));
      if (targetTab) selectTab(targetTab);
    });
  });

  // Fallback arrow-key navigation when native focusgroup is unavailable
  if (!Object.hasOwn(HTMLElement.prototype, 'focusGroup')) {
    tabs.forEach((t, idx) => {
      t.tabIndex = idx === 0 ? 0 : -1;
    });
    tablist.addEventListener('keydown', (e) => {
      const currentIdx = tabs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
      if (currentIdx === -1) return;
      let nextIdx = null;
      if (e.key === 'ArrowRight') nextIdx = (currentIdx + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') nextIdx = (currentIdx - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') nextIdx = 0;
      else if (e.key === 'End') nextIdx = tabs.length - 1;
      if (nextIdx !== null) {
        e.preventDefault();
        selectTab(tabs[nextIdx], true);
      }
    });
  }

  // Scrollability fade fallback when scroll-driven animations are unsupported
  if (!CSS.supports('animation-timeline', 'scroll()') && tablist) {
    const startSentinel = document.createElement('div');
    const endSentinel = document.createElement('div');
    [startSentinel, endSentinel].forEach((el) => {
      el.style.cssText = 'min-inline-size: 1px; min-block-size: 1px; flex: none;';
      el.setAttribute('aria-hidden', 'true');
    });
    tablist.prepend(startSentinel);
    tablist.append(endSentinel);

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.target === startSentinel) {
            tablist.classList.toggle('more-start', !entry.isIntersecting);
          } else if (entry.target === endSentinel) {
            tablist.classList.toggle('more-end', !entry.isIntersecting);
          }
        }
      },
      { root: tablist }
    );
    observer.observe(startSentinel);
    observer.observe(endSentinel);
  }

  // Light-dismiss fallback for browsers without <dialog closedby="any">
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    inspectDialog.addEventListener('click', (event) => {
      if (event.target !== inspectDialog) return;
      const rect = inspectDialog.getBoundingClientRect();
      const isDialogContent =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (isDialogContent) return;
      inspectDialog.close();
    });
  }

  function renderEvidenceList(items, fallbackText) {
    if (!items || items.length === 0) {
      return `<p style="color: var(--text-muted)">${escapeHtml(fallbackText)}</p>`;
    }
    return `<ul class="evidence-list">${items.map((x) => `<li>${escapeHtml(x)}</li>`).join('')}</ul>`;
  }

  function openInspector(item) {
    const info = item.file_info || {};
    const layers = item.layers || {};
    const l1 = layers.c2pa || {};
    const l2 = layers.pixel_watermark || {};
    const l3 = layers.container || {};
    const l4 = layers.dsp_carrier || {};
    const visuals = l4.visuals || {};
    const cameraExif = l3.metadata_found && l3.metadata_found.camera_exif_present;

    inspectDialogTitle.textContent = item.filename;
    inspectDialogSubtitle.textContent = `${info.format || '?'} • ${info.width || '?'}×${info.height || '?'} px • ${info.mode || '?'} • ${info.file_size_bytes || 0} bytes`;

    panelOverviewContent.innerHTML = `
      <div>${verdictBadge(item.verdict)} &nbsp; <strong>${escapeHtml(item.platform || 'Unidentified')}</strong></div>
      <div class="kv-grid">
        <div class="kv-box"><span>Triggered Layers</span><strong>${escapeHtml((item.triggered_layers || []).join(', ') || 'None')}</strong></div>
        <div class="kv-box"><span>SHA-256</span><strong>${escapeHtml((info.sha256 || '').slice(0, 20))}...</strong></div>
      </div>
      <h3 style="margin:1rem 0 0.25rem; font-size:0.95rem;">Findings</h3>
      ${renderEvidenceList(item.all_evidence, 'No watermark signatures or DSP carrier signals detected.')}
      <div class="visuals-grid">
        <div class="visual-card">
          <h4>2D FFT Spectrum</h4>
          ${visuals.fft_spectrum_b64 ? `<img src="${visuals.fft_spectrum_b64}" alt="2D FFT Spectrum">` : '<p>Unavailable</p>'}
        </div>
        <div class="visual-card">
          <h4>LSB Bit-Plane</h4>
          ${visuals.lsb_plane_b64 ? `<img src="${visuals.lsb_plane_b64}" alt="LSB Bit-Plane">` : '<p>Unavailable</p>'}
        </div>
        <div class="visual-card">
          <h4>High-Pass Residual (20x)</h4>
          ${visuals.residual_map_b64 ? `<img src="${visuals.residual_map_b64}" alt="High-Pass Residual">` : '<p>Unavailable</p>'}
        </div>
      </div>
    `;

    panelC2paContent.innerHTML = `
      <p><strong>Status:</strong> ${l1.detected ? 'Detected' : 'None'}</p>
      ${renderEvidenceList(l1.evidence, 'No C2PA / JUMBF container found.')}
      <pre class="code-block">${escapeHtml(JSON.stringify(l1.manifest_summary || {}, null, 2))}</pre>
    `;

    panelPixelContent.innerHTML = `
      <p><strong>Status:</strong> ${l2.detected ? 'Decoded' : 'None'}</p>
      <div class="kv-grid">
        <div class="kv-box"><span>Method</span><strong>${escapeHtml((l2.methods_matched || []).join(', ') || '-')}</strong></div>
        <div class="kv-box"><span>Block Consistency</span><strong>${l2.bit_consistency != null ? (l2.bit_consistency * 100).toFixed(1) + '%' : '-'}</strong></div>
      </div>
      ${renderEvidenceList(l2.evidence, 'No embedded pixel or frequency bitstream decoded.')}
      ${l2.decoded_payload ? `<h4 style="margin:0.9rem 0 0.25rem;">Decoded Payload</h4><pre class="code-block">${escapeHtml(l2.decoded_payload)}</pre>` : ''}
      <pre class="code-block">${escapeHtml(JSON.stringify(l2.details || {}, null, 2))}</pre>
    `;

    panelContainerContent.innerHTML = `
      <p><strong>Status:</strong> ${l3.detected ? 'Detected' : 'None'}</p>
      <div class="kv-grid">
        <div class="kv-box"><span>Camera EXIF</span><strong>${cameraExif ? 'Present' : 'Absent'}</strong></div>
        <div class="kv-box"><span>XMP Packets</span><strong>${l3.metadata_found ? (l3.metadata_found.xmp_packets_count || 0) : 0}</strong></div>
      </div>
      ${renderEvidenceList(l3.evidence, 'No generator signatures found in metadata.')}
      <h4 style="margin:0.9rem 0 0.25rem;">Chunks &amp; Metadata</h4>
      <pre class="code-block">${escapeHtml(JSON.stringify({ chunks: l3.chunks || [], metadata: l3.metadata_found || {} }, null, 2))}</pre>
    `;

    panelDspContent.innerHTML = `
      <p><strong>Status:</strong> ${l4.detected ? 'Carrier Detected' : 'Nominal'}</p>
      <p><strong>Carrier Type:</strong> ${escapeHtml(l4.carrier_type || '-')}</p>
      ${renderEvidenceList(l4.evidence, 'No periodic spectral peaks or radial ring modulation detected.')}
      <pre class="code-block">${escapeHtml(JSON.stringify(l4.metrics || {}, null, 2))}</pre>
    `;

    selectTab(tabs[0]);
    inspectDialog.showModal();
  }

  resultsTbody.addEventListener('click', (event) => {
    const btn = event.target.closest('.inspect-btn');
    if (!btn) return;
    const item = currentResults[Number(btn.dataset.index)];
    if (item) openInspector(item);
  });

  pickFilesBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    analyzeFiles(e.target.files);
    fileInput.value = '';
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-active');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files) {
      analyzeFiles(e.dataTransfer.files);
    }
  });

  loadSamplesBtn.addEventListener('click', loadDemoSamples);

  clearResultsBtn.addEventListener('click', () => {
    currentResults = [];
    renderTable();
    hideStatus();
  });

  exportJsonBtn.addEventListener('click', () => {
    const cleanReport = currentResults.map((r) => {
      const clone = structuredClone(r);
      if (clone.layers && clone.layers.dsp_carrier) {
        clone.layers.dsp_carrier.visuals = {};
      }
      return clone;
    });
    const blob = new Blob([JSON.stringify(cleanReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'watermark_inspection_report.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  exportCsvBtn.addEventListener('click', () => {
    const cols = ['filename', 'verdict', 'platform', 'triggered_layers', 'width', 'height', 'format', 'sha256', 'evidence'];
    const lines = [cols.join(',')];
    for (const r of currentResults) {
      const info = r.file_info || {};
      const row = [
        r.filename || '',
        r.verdict || '',
        r.platform || '',
        (r.triggered_layers || []).join(' | '),
        info.width || '',
        info.height || '',
        info.format || '',
        info.sha256 || '',
        (r.all_evidence || []).join(' ; '),
      ].map((v) => `"${String(v).replace(/"/g, '""')}"`);
      lines.push(row.join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'watermark_inspection_report.csv';
    a.click();
    URL.revokeObjectURL(url);
  });

  renderTable();
})();
