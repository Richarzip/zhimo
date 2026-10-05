const state = {
  file: null,
  mode: 'single',
};

const $ = (id) => document.getElementById(id);

function setText(id, text) {
  $(id).textContent = text ?? '-';
}

function formatBool(v) {
  return v ? '存在' : '缺失';
}

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const status = $('serverStatus');
    status.textContent = data.model_exists ? '服务就绪' : '缺少权重';
    status.className = `status-pill ${data.model_exists ? 'ok' : 'warn'}`;
    $('envList').innerHTML = `
      <dt>模型权重</dt><dd>${formatBool(data.model_exists)}</dd>
      <dt>权重路径</dt><dd>${escapeHtml(data.model_path)}</dd>
      <dt>Chroma</dt><dd>${formatBool(data.chroma_db_exists)}</dd>
      <dt>Python</dt><dd>${escapeHtml(data.python)}</dd>
      <dt>样例图</dt><dd>${escapeHtml((data.sample_images || []).join(', ') || '-')}</dd>
    `;
  } catch (err) {
    $('serverStatus').textContent = '服务异常';
    $('serverStatus').className = 'status-pill warn';
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function setupUpload() {
  const input = $('imageInput');
  const dropzone = $('dropzone');
  input.addEventListener('change', () => setFile(input.files?.[0]));
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files?.[0]));
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  const url = URL.createObjectURL(file);
  $('preview').src = url;
  $('previewWrap').classList.remove('hidden');
  $('analyzeBtn').disabled = false;
}

function setupModeButtons() {
  document.querySelectorAll('.segmented button').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.segmented button').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      state.mode = button.dataset.mode;
      $('camToggle').disabled = state.mode === 'multi';
      if (state.mode === 'multi') $('camToggle').checked = false;
    });
  });
}

async function analyze() {
  if (!state.file) return;
  $('analyzeBtn').disabled = true;
  $('analyzeBtn').textContent = 'Agent 调用中...';
  $('runState').textContent = '运行中';
  renderTimeline([{ name: 'request', status: 'running', detail: '正在发送图片到本地服务' }]);
  clearResult();

  const form = new FormData();
  form.append('image', state.file);
  form.append('mode', state.mode);
  form.append('rag', String($('ragToggle').checked));
  form.append('cam', String($('camToggle').checked));
  form.append('tta', String($('ttaToggle').checked));
  form.append('prompt', $('prompt').value);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();
    renderResult(data);
    $('runState').textContent = data.blocked ? '已停止在可诊断节点' : '完成';
  } catch (err) {
    const data = { error: 'browser_error', diagnostic: { message: '浏览器请求失败', raw: String(err) } };
    renderResult(data);
    $('runState').textContent = '失败';
  } finally {
    $('analyzeBtn').disabled = false;
    $('analyzeBtn').textContent = '开始 Agent 调用';
  }
}

function clearResult() {
  setText('calligrapher', '-');
  setText('confidence', '-');
  setText('reliability', '-');
  $('summary').textContent = '正在分析...';
  $('diagnostic').classList.add('hidden');
  $('qualityGrid').innerHTML = '';
  $('rankList').innerHTML = '';
  $('camFigure').classList.add('hidden');
  $('segFigure').classList.add('hidden');
  $('rawJson').textContent = '{}';
}

function renderResult(data) {
  renderTimeline(data.steps || []);
  $('rawJson').textContent = JSON.stringify(data, null, 2);
  $('resultMode').textContent = data.mode === 'multi' ? '多字作品' : data.mode === 'single' ? '单字' : '异常';
  $('summary').textContent = data.summary || data.diagnostic?.message || data.message || '没有返回摘要。';

  const diagnostic = data.diagnostic || data.knowledge_diagnostic;
  if (diagnostic) {
    $('diagnostic').classList.remove('hidden');
    $('diagnostic').innerHTML = `
      <strong>${escapeHtml(diagnostic.message || '诊断信息')}</strong><br>
      <span>${escapeHtml(diagnostic.suggestion || '')}</span>
      ${diagnostic.raw ? `<pre>${escapeHtml(diagnostic.raw)}</pre>` : ''}
    `;
  }

  const rec = data.recognition || {};
  setText('calligrapher', rec.calligrapher || '-');
  setText('confidence', rec.confidence ?? '-');
  setText('reliability', data.reliability || rec.consistency || '-');
  renderQuality(data.quality || {});
  renderRanks(rec);

  const heatmap = rec.evidence?.heatmap;
  if (heatmap) {
    $('camImage').src = heatmap.startsWith('data:') ? heatmap : `data:image/png;base64,${heatmap}`;
    $('camFigure').classList.remove('hidden');
  }
  if (data.segmentation?.overlay) {
    $('segImage').src = data.segmentation.overlay;
    $('segFigure').classList.remove('hidden');
  }
}

function renderTimeline(steps) {
  if (!steps.length) {
    $('timeline').innerHTML = '<li><div class="step-detail">暂无步骤。</div></li>';
    return;
  }
  $('timeline').innerHTML = steps.map((step) => `
    <li class="${escapeHtml(step.status || '')}">
      <div class="step-head">
        <span class="dot"></span>
        <span class="step-name">${escapeHtml(step.name)}</span>
      </div>
      <div class="step-detail">${escapeHtml(step.detail || '')}</div>
      ${step.error ? `<div class="step-error">${escapeHtml(step.error)}</div>` : ''}
    </li>
  `).join('');
}

function renderQuality(quality) {
  const labels = [
    ['yellow', '泛黄'],
    ['fade', '褪色'],
    ['noise', '噪声'],
    ['blur', '模糊'],
    ['overall', '综合'],
  ];
  $('qualityGrid').innerHTML = labels.map(([key, label]) => {
    const value = Number(quality[key] ?? 0);
    const pct = Math.max(0, Math.min(1, value)) * 100;
    return `
      <div class="bar-row">
        <span>${label}</span>
        <span class="bar"><i style="width:${pct}%"></i></span>
        <span>${Number.isFinite(value) ? value.toFixed(3) : '-'}</span>
      </div>
    `;
  }).join('');
}

function renderRanks(rec) {
  let items = [];
  if (Array.isArray(rec.top_k)) {
    items = rec.top_k.map((item) => [item.name, item.confidence]);
  } else if (rec.vote_distribution) {
    items = Object.entries(rec.vote_distribution);
  }
  if (!items.length && Array.isArray(rec.per_char_results)) {
    items = rec.per_char_results.map((item, index) => [`#${index + 1} ${item.name}`, item.confidence]);
  }
  $('rankList').innerHTML = items.length ? items.map(([name, score]) => `
    <div class="rank-item"><span>${escapeHtml(name)}</span><strong>${escapeHtml(score)}</strong></div>
  `).join('') : '<div class="muted">暂无候选。</div>';
}

function resetAll() {
  state.file = null;
  $('imageInput').value = '';
  $('preview').src = '';
  $('previewWrap').classList.add('hidden');
  $('analyzeBtn').disabled = true;
  $('runState').textContent = '等待输入';
  renderTimeline([]);
  clearResult();
  $('summary').textContent = '上传图片后开始分析。即使缺少权重，本页面也会展示流程已执行到模型推理入口。';
}

function boot() {
  setupUpload();
  setupModeButtons();
  $('analyzeBtn').addEventListener('click', analyze);
  $('resetBtn').addEventListener('click', resetAll);
  renderTimeline([]);
  loadHealth();
}

boot();
