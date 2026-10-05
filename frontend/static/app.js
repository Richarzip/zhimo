const state = {
  file: null,
  mode: 'single',
  previewUrl: null,
  fileVersion: 0,
  requestId: 0,
  controller: null,
  analyzing: false,
  validating: false,
};

const $ = (id) => document.getElementById(id);
const MAX_UPLOAD_BYTES = 32 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 25_000_000;
const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/x-ms-bmp', 'image/gif']);

function setText(id, text) {
  $(id).textContent = text ?? '-';
}

function formatBool(v) {
  return v ? '存在' : '缺失';
}

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
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

function updateAnalyzeButton() {
  $('analyzeBtn').disabled = !state.file || state.analyzing || state.validating;
  $('analyzeBtn').textContent = state.analyzing ? 'Agent 调用中...' : '开始 Agent 调用';
}

function cancelAnalysis() {
  // Aborting fetch alone cannot prevent an already-resolved response from rendering.
  state.requestId += 1;
  state.controller?.abort();
  state.controller = null;
  state.analyzing = false;
  updateAnalyzeButton();
}

function clearFile() {
  state.file = null;
  $('imageInput').value = '';
  $('preview').removeAttribute('src');
  $('previewWrap').classList.add('hidden');
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
}

function showUploadError(message = '') {
  $('uploadError').textContent = message;
  $('uploadError').classList.toggle('hidden', !message);
  $('imageInput').setAttribute('aria-invalid', String(Boolean(message)));
}

function checkImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (!image.naturalWidth || !image.naturalHeight) {
        reject(new Error('图片无法解码，请选择有效图片。'));
      } else if (image.naturalWidth * image.naturalHeight > MAX_IMAGE_PIXELS) {
        reject(new Error('图片不能超过 2500 万像素，请缩小后再上传。'));
      } else {
        resolve();
      }
    };
    image.onerror = () => reject(new Error('图片无法解码，请选择有效图片。'));
    image.src = url;
  });
}

async function setFile(file) {
  if (!file) return;
  const version = ++state.fileVersion;
  cancelAnalysis();
  clearFile();
  showUploadError();
  state.validating = true;
  updateAnalyzeButton();
  renderTimeline([]);
  clearResult('正在检查图片...');
  $('runState').textContent = '检查图片中';
  let url;
  try {
    const supported = file.type ? IMAGE_TYPES.has(file.type.toLowerCase()) : /\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name);
    if (!supported) throw new Error('请选择 JPEG、PNG、WEBP、BMP 或 GIF 图片。');
    if (!file.size) throw new Error('文件为空，请选择有效图片。');
    if (file.size > MAX_UPLOAD_BYTES) throw new Error('图片不能超过 32 MiB，请压缩后再上传。');
    url = URL.createObjectURL(file);
    await checkImage(url);
    if (version !== state.fileVersion) return;
    state.file = file;
    state.previewUrl = url;
    $('preview').src = url;
    $('previewWrap').classList.remove('hidden');
    $('runState').textContent = '等待分析';
    $('summary').textContent = '图片已就绪，点击开始分析。';
  } catch (err) {
    if (version !== state.fileVersion) return;
    showUploadError(err.message);
    $('runState').textContent = '等待输入';
    $('summary').textContent = '请重新选择有效图片。';
  } finally {
    if (url && url !== state.previewUrl) URL.revokeObjectURL(url);
    if (version === state.fileVersion) {
      state.validating = false;
      updateAnalyzeButton();
    }
  }
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
  if (!state.file || state.analyzing || state.validating) return;
  const requestId = ++state.requestId;
  const controller = new AbortController();
  state.controller = controller;
  state.analyzing = true;
  updateAnalyzeButton();
  $('runState').textContent = '运行中';
  renderTimeline([{ name: 'request', status: 'running', detail: '正在发送图片到本地服务' }]);
  clearResult('正在分析...');

  try {
    const form = new FormData();
    form.append('image', state.file);
    form.append('mode', state.mode);
    form.append('rag', String($('ragToggle').checked));
    form.append('cam', String($('camToggle').checked));
    form.append('tta', String($('ttaToggle').checked));
    form.append('prompt', $('prompt').value);

    const res = await fetch('/api/analyze', { method: 'POST', body: form, signal: controller.signal });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error(`服务返回了无法解析的响应（HTTP ${res.status}）。`);
    }
    if (requestId !== state.requestId) return;
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw new Error(`服务返回的数据格式不正确（HTTP ${res.status}）。`);
    }
    if (!res.ok || data.error) {
      renderResult({ ...data, message: data.message || `请求失败（HTTP ${res.status}）。` });
      $('runState').textContent = '失败';
      return;
    }
    renderResult(data);
    const partial = data.knowledge_diagnostic || data.recognition?.evidence?.error || data.steps?.some((step) => step.status === 'error');
    $('runState').textContent = data.blocked ? '已停止在可诊断节点' : partial ? '部分完成' : '完成';
  } catch (err) {
    if (requestId !== state.requestId || err.name === 'AbortError') return;
    const data = { error: 'browser_error', diagnostic: { message: '浏览器请求失败', raw: String(err) } };
    renderResult(data);
    $('runState').textContent = '失败';
  } finally {
    if (requestId === state.requestId) {
      state.controller = null;
      state.analyzing = false;
      updateAnalyzeButton();
    }
  }
}

function clearResult(summary = '上传图片后开始分析。') {
  setText('calligrapher', '-');
  setText('confidence', '-');
  setText('reliability', '-');
  $('summary').textContent = summary;
  $('resultMode').textContent = '未运行';
  $('diagnostic').classList.add('hidden');
  $('diagnostic').innerHTML = '';
  $('qualityGrid').innerHTML = '';
  $('rankList').innerHTML = '';
  $('camFigure').classList.add('hidden');
  $('segFigure').classList.add('hidden');
  $('camImage').removeAttribute('src');
  $('segImage').removeAttribute('src');
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
    const value = quality[key];
    const valid = typeof value === 'number' && Number.isFinite(value);
    const pct = valid ? Math.max(0, Math.min(1, value)) * 100 : 0;
    return `
      <div class="bar-row">
        <span>${label}</span>
        <span class="bar"><i style="width:${pct}%"></i></span>
        <span>${valid ? value.toFixed(3) : '未评估'}</span>
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
  state.fileVersion += 1;
  cancelAnalysis();
  clearFile();
  state.validating = false;
  showUploadError();
  updateAnalyzeButton();
  $('runState').textContent = '等待输入';
  renderTimeline([]);
  clearResult();
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
