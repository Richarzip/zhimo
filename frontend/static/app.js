const state = {
  file: null,
  mode: 'auto',
  camBeforeMulti: null,
  previewUrl: null,
  fileVersion: 0,
  requestId: 0,
  controller: null,
  analyzing: false,
  validating: false,
  exporting: false,
  exportPdf: false,
  sessionId: newSessionId(),
  messages: [], // 对话记录 [{role, text, previewUrl, loading, data}]
};

function newSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 's-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
}

const $ = (id) => document.getElementById(id);
const MAX_UPLOAD_BYTES = 32 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 25_000_000;
const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/x-ms-bmp', 'image/gif']);

const CHAT_EMPTY_HTML =
  '<div class="chat-empty" id="chatEmpty">上传书法图片或输入问题，与鉴赏 Agent 多轮对话。Agent 能记住同一会话中的上下文。</div>';

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
    const modelPaths = (data.model_paths?.length ? data.model_paths : [data.model_path])
      .filter(Boolean)
      .map(escapeHtml)
      .join('<br>');
    $('envList').innerHTML = `
      <dt>模型权重</dt><dd>${formatBool(data.model_exists)}</dd>
      <dt>权重路径</dt><dd>${modelPaths}</dd>
      <dt>Chroma</dt><dd>${formatBool(data.chroma_db_exists)}</dd>
      <dt>Python</dt><dd>${escapeHtml(data.python)}</dd>
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
  const canSend = Boolean(state.file) || $('prompt').value.trim();
  $('analyzeBtn').disabled = !canSend || state.analyzing || state.validating;
  $('analyzeBtn').textContent = state.analyzing ? '对话中...' : '发送';
}

function cancelAnalysis() {
  // Aborting fetch alone cannot prevent an already-resolved response from rendering.
  completeAiMessage(state.requestId, { error: '请求已取消。' });
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
  const previousUrl = state.previewUrl;
  state.previewUrl = null;
  releasePreviewUrl(previousUrl);
  updatePdfToggle();
}

function updatePdfToggle() {
  const toggle = $('pdfToggle');
  if (!toggle) return;
  toggle.disabled = !state.file;
  if (!state.file) toggle.checked = false;
}

function releasePreviewUrl(url) {
  if (url && url !== state.previewUrl && !state.messages.some((message) => message.previewUrl === url)) {
    URL.revokeObjectURL(url);
  }
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
    updatePdfToggle();
    state.previewUrl = url;
    $('preview').src = url;
    $('previewWrap').classList.remove('hidden');
    $('runState').textContent = '等待分析';
  } catch (err) {
    if (version !== state.fileVersion) return;
    showUploadError(err.message);
    $('runState').textContent = '等待输入';
  } finally {
    releasePreviewUrl(url);
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
      const wasMulti = state.mode === 'multi';
      state.mode = button.dataset.mode;
      const cam = $('camToggle');
      cam.disabled = state.mode === 'multi';
      if (state.mode === 'multi' && !wasMulti) {
        state.camBeforeMulti = cam.checked;
        cam.checked = false;
      } else if (wasMulti && state.mode !== 'multi') {
        cam.checked = state.camBeforeMulti;
        state.camBeforeMulti = null;
      }
    });
  });
}

async function analyze() {
  if (state.analyzing || state.validating) return;
  const text = $('prompt').value.trim();
  if (!state.file && !text) return;
  const requestId = ++state.requestId;
  const controller = new AbortController();
  state.controller = controller;
  state.analyzing = true;
  updateAnalyzeButton();
  $('runState').textContent = '运行中';
  // 结果栏保留上一轮最新非空数据，等待响应后由 renderResult 决定是否刷新；
  // 仅本轮 Agent 过程时间线立即更新为“发送中”。
  renderTimeline([{ name: 'request', status: 'running', detail: '正在发送消息到本地服务' }]);

  // 用户消息入对话流 + AI loading 占位
  pushUserMessage(text, state.previewUrl);
  pushAiMessage({ loading: true, requestId });

  try {
    const form = new FormData();
    form.append('mode', 'chat');
    form.append('analysis_mode', state.mode);
    form.append('session_id', state.sessionId);
    form.append('text', text);
    if (state.file) form.append('image', state.file);
    form.append('rag', String($('ragToggle').checked));
    form.append('cam', String($('camToggle').checked));
    form.append('tta', String($('ttaToggle').checked));
    form.append('denoise', String($('denoiseToggle').checked));
    form.append('examples', String($('examplesToggle').checked));
    const autoExportPdf = Boolean(state.file && $('pdfToggle')?.checked);
    form.append('pdf', String(autoExportPdf));
    form.append('prompt', text);

    const fetchPromise = fetch('/api/analyze', { method: 'POST', body: form, signal: controller.signal });
    // 文本与图片均已随请求发出：立即清空输入栏，不等推理结束
    clearFile();
    $('prompt').value = '';
    const res = await fetchPromise;
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
      const message = data.message || `请求失败（HTTP ${res.status}）。`;
      renderResult({ ...data, message });
      completeAiMessage(requestId, { error: message });
      $('runState').textContent = '失败';
      return;
    }
    // 对话回复 + 右侧最新结果区（识别相关摘要由 renderResult 处理）
    completeAiMessage(requestId, { text: data.reply || '', data });
    renderResult(data);
    // 导出模式：拿到 PDF 后自动触发下载
    if (autoExportPdf && data.pdf?.data_url) {
      const link = document.createElement('a');
      link.href = data.pdf.data_url;
      link.download = data.pdf.filename || 'zhimo_report.pdf';
      link.click();
    }
    $('prompt').value = '';
    const partial = data.knowledge_diagnostic || data.recognition?.evidence?.error || data.steps?.some((step) => step.status === 'error');
    $('runState').textContent = data.blocked ? '已停止在可诊断节点' : partial ? '部分完成' : '完成';
  } catch (err) {
    if (requestId !== state.requestId) return;
    if (err.name === 'AbortError') {
      completeAiMessage(requestId, { error: '请求已取消。' });
      $('runState').textContent = '已取消';
      return;
    }
    const data = { error: 'browser_error', diagnostic: { message: '浏览器请求失败', raw: String(err) } };
    completeAiMessage(requestId, { error: '浏览器请求失败' });
    renderResult(data);
    $('runState').textContent = '失败';
  } finally {
    if (requestId === state.requestId) {
      state.exportPdf = false;
      const pdfToggle = $('pdfToggle');
      if (pdfToggle) {
        pdfToggle.checked = false;
        pdfToggle.disabled = true;
      }
      const pdfBtn = $('exportPdfBtn');
      if (pdfBtn) pdfBtn.textContent = '导出为PDF';
      state.controller = null;
      state.analyzing = false;
      updateAnalyzeButton();
    }
  }
}

function clearResult() {
  setText('calligrapher', '-');
  setText('confidence', '-');
  setText('reliability', '-');
  $('resultMode').textContent = '未运行';
  $('diagnostic').classList.add('hidden');
  $('diagnostic').innerHTML = '';
  $('qualityGrid').innerHTML = '';
  $('rankList').innerHTML = '';
  $('segFigure').classList.add('hidden');
  $('referenceBox').classList.add('hidden');
  $('referenceGrid').innerHTML = '';
  $('referenceNote').textContent = '';
  $('reportActions').classList.add('hidden');
  $('downloadPdfBtn').onclick = null;
  $('segImage').removeAttribute('src');
  $('rawJson').textContent = '{}';
}

// ====================== 多轮对话渲染 ======================

function pushUserMessage(text, previewUrl) {
  state.messages.push({ role: 'user', text: text || '', previewUrl: previewUrl || null });
  renderChat();
}

function pushAiMessage({ text = '', loading = false, data = null, error = '', requestId = null } = {}) {
  state.messages.push({ role: 'ai', text, loading, data, error, requestId });
  renderChat();
}

function completeAiMessage(requestId, { text = '', data = null, error = '' } = {}) {
  const message = state.messages.find((item) => item.role === 'ai' && item.requestId === requestId && item.loading);
  if (!message) return;
  Object.assign(message, { text, data, error, loading: false });
  renderChat();
}

function renderChat() {
  const list = $('chatList');
  if (!list) return;
  if (!state.messages.length) {
    list.innerHTML = CHAT_EMPTY_HTML;
    return;
  }
  list.innerHTML = state.messages.map((m) => {
    if (m.role === 'user') {
      let body = '';
      if (m.text) body += `<p class="chat-text">${escapeHtml(m.text)}</p>`;
      if (m.previewUrl) body += `<img class="chat-img" src="${m.previewUrl}" alt="用户上传的图片">`;
      return `<div class="chat-msg user"><div class="chat-bubble">${body || '<p class="chat-text">（图片）</p>'}</div></div>`;
    }
    let body;
    if (m.loading) {
      body = '<div class="chat-content chat-loading"><span></span><span></span><span></span></div>';
    } else if (m.error) {
      body = `<div class="chat-content chat-error">${escapeHtml(m.error)}</div>`;
    } else {
      body = `<div class="chat-content">${renderMarkdown(m.text)}</div>`;
      body += chatCardsHtml(m.data || {});
    }
    return `<div class="chat-msg ai"><div class="chat-avatar ai">智墨</div><div class="chat-bubble">${body}</div></div>`;
  }).join('');
  scrollChat();
}

function scrollChat() {
  const list = $('chatList');
  if (list) list.scrollTop = list.scrollHeight;
}

function chatCardsHtml(data) {
  const rec = data.recognition || {};
  const cards = [];
  if (rec.calligrapher) {
    cards.push(`<div class="chat-metric"><span>书法家</span><strong>${escapeHtml(rec.calligrapher)}</strong></div>`);
  }
  if (rec.confidence !== undefined && rec.confidence !== null) {
    cards.push(`<div class="chat-metric"><span>置信度</span><strong>${escapeHtml(String(rec.confidence))}</strong></div>`);
  }
  if (data.reliability) {
    cards.push(`<div class="chat-metric"><span>可靠性</span><strong>${escapeHtml(String(data.reliability))}</strong></div>`);
  }
  const parts = [];
  if (cards.length) parts.push(`<div class="chat-cards">${cards.join('')}</div>`);
  if (data.inversion?.was_inverted) {
    parts.push('<div class="chat-note">检测到黑底白字拓印，已自动反转为白底黑字后再识别。</div>');
  }
  return parts.join('');
}

// Markdown 渲染（markdown-it，本地 vendor 库；库未加载时降级为纯文本）
const zhimoMd = (typeof window !== 'undefined' && window.markdownit)
  ? window.markdownit({ html: false, linkify: false })
  : null;

function renderMarkdown(text) {
  if (zhimoMd) return zhimoMd.render(String(text || ''));
  return escapeHtml(String(text || ''));
}

function clearChat() {
  const historyUrls = new Set(state.messages.map((message) => message.previewUrl).filter(Boolean));
  state.messages = [];
  $('chatList').innerHTML = CHAT_EMPTY_HTML;
  historyUrls.forEach(releasePreviewUrl);
}

// 重置会话记忆（保留已上传的图片与选项；当前无 UI 入口，供内部逻辑与测试使用）
function resetChat() {
  state.sessionId = newSessionId();
  state.fileVersion += 1;
  cancelAnalysis();
  state.validating = false;
  $('prompt').value = '';
  updateAnalyzeButton();
  $('runState').textContent = '等待输入';
  renderTimeline([]);
  clearResult();
  clearChat();
}

function renderResult(data) {
  // 只有本轮真正携带识别/分析产物时才刷新结果区；
  // 纯对话或错误轮次保持上一轮的最新非空数据，避免结果栏被清空。
  // 注意用非空判断：后端可能返回空对象 {}，存在性判断会误判为“有结果”。
  const hasRecognition = Boolean(
    (data.recognition && Object.keys(data.recognition).length > 0) ||
    (data.quality && Object.keys(data.quality).length > 0) ||
    (data.segmentation && data.segmentation.overlay) ||
    (data.pdf && data.pdf.data_url)
  );
  if (hasRecognition) clearResult();

  renderTimeline(data.steps || []);
  $('rawJson').textContent = JSON.stringify(data, null, 2);
  $('resultMode').textContent = data.mode === 'multi' ? '多字作品' : data.mode === 'single' ? '单字' : data.mode === 'chat' ? '对话' : '异常';

  const diagnostic = data.diagnostic || data.knowledge_diagnostic;
  if (diagnostic) {
    $('diagnostic').classList.remove('hidden');
    $('diagnostic').innerHTML = `
      <strong>${escapeHtml(diagnostic.message || '诊断信息')}</strong><br>
      <span>${escapeHtml(diagnostic.suggestion || '')}</span>
      ${diagnostic.raw ? `<pre>${escapeHtml(diagnostic.raw)}</pre>` : ''}
    `;
  } else if (hasRecognition) {
    // 有识别数据但无诊断：清掉上一轮残留诊断（原 clearResult 行为）
    $('diagnostic').classList.add('hidden');
    $('diagnostic').innerHTML = '';
  }

  // 纯对话/错误轮次：识别区保持上一轮数据，不再刷新
  if (!hasRecognition) return;

  const rec = data.recognition || {};
  setText('calligrapher', rec.calligrapher || '-');
  setText('confidence', rec.confidence ?? '-');
  setText('reliability', data.reliability || rec.consistency || '-');
  renderQuality(data.quality || {});
  renderRanks(rec);

  if (data.segmentation?.overlay) {
    $('segImage').src = data.segmentation.overlay;
    $('segFigure').classList.remove('hidden');
  }
  renderReferences(data);
  if (data.pdf?.data_url) {
    $('reportActions').classList.remove('hidden');
    $('downloadPdfBtn').onclick = () => {
      const link = document.createElement('a');
      link.href = data.pdf.data_url;
      link.download = data.pdf.filename || 'zhimo_report.pdf';
      link.click();
    };
  }
}

function renderReferences(data) {
  const examples = Array.isArray(data.reference_examples) ? data.reference_examples : [];
  const note = data.reference_examples_note || '';
  if (!examples.length && !note) return;
  $('referenceBox').classList.remove('hidden');
  $('referenceNote').textContent = note;
  $('referenceGrid').innerHTML = examples.map((item) => `
    <figure class="reference-item">
      <img src="${escapeHtml(item.url)}" alt="${escapeHtml(item.title || 'calligraphy example')}" />
      <figcaption>
        <strong>${escapeHtml(item.title || 'work example')}</strong>
        <small>${escapeHtml(item.license || 'license unavailable')}</small>
        <a href="${escapeHtml(item.source_url || '#')}" target="_blank" rel="noreferrer">Source</a>
      </figcaption>
    </figure>
  `).join('');
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
  state.sessionId = newSessionId();
  state.fileVersion += 1;
  cancelAnalysis();
  clearFile();
  state.validating = false;
  showUploadError();
  $('prompt').value = '';
  updateAnalyzeButton();
  $('runState').textContent = '等待输入';
  renderTimeline([]);
  clearResult();
  clearChat();
}

function boot() {
  setupUpload();
  setupModeButtons();
  $('analyzeBtn').addEventListener('click', analyze);
  $('resetBtn').addEventListener('click', resetAll);
  $('prompt').addEventListener('input', updateAnalyzeButton);
  renderTimeline([]);
  clearChat();
  loadHealth();
}

boot();

// ===== 结果栏折叠（独立功能，不影响其他逻辑）=====
// 测试环境（Node）没有 localStorage，统一走安全封装
function zhimoStorage() {
  try {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    return null; // 隐私模式等场景访问可能抛异常
  }
}

function setupResultToggle() {
  const btn = $('resultToggleBtn');
  if (!btn) return;
  const storage = zhimoStorage();
  // 刷新后恢复上次的隐藏状态
  if (storage && storage.getItem('zhimoResultHidden') === '1') {
    document.body.classList.add('result-hidden');
    btn.textContent = '显示结果栏';
  }
  btn.addEventListener('click', () => {
    const hidden = document.body.classList.toggle('result-hidden');
    btn.textContent = hidden ? '显示结果栏' : '隐藏结果栏';
    if (storage) storage.setItem('zhimoResultHidden', hidden ? '1' : '0');
  });
}

setupResultToggle();

// ===== 导出对话为 PDF =====

// markdown 回复 → 纯文本（剥离 **、#、列表等标记，避免画进 PDF）
function markdownToPlain(text) {
  if (!text) return '';
  try {
    const html = window.markdownit({ html: false, linkify: false }).render(String(text));
    const div = document.createElement('div');
    div.innerHTML = html;
    return (div.textContent || '').replace(/\s+/g, ' ').trim();
  } catch {
    return String(text).replace(/[*#`>\[\]()!-]/g, '').trim();
  }
}

// 对话里的图片（blob URL）→ 压缩为 <=maxSide 的 JPEG data URL，避免请求体过大
function imageToDataUrl(url, maxSide = 600) {
  return fetch(url)
    .then((response) => response.blob())
    .then((blob) => {
      if (typeof createImageBitmap === 'function') {
        return createImageBitmap(blob).then((bmp) => {
          const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
          const canvas = document.createElement('canvas');
          canvas.width = Math.max(1, Math.round(bmp.width * scale));
          canvas.height = Math.max(1, Math.round(bmp.height * scale));
          canvas.getContext('2d').drawImage(bmp, 0, 0, canvas.width, canvas.height);
          bmp.close();
          return canvas.toDataURL('image/jpeg', 0.85);
        });
      }
      const img = new Image();
      img.src = url;
      return new Promise((resolve, reject) => {
        img.onload = () => {
          const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
          const canvas = document.createElement('canvas');
          canvas.width = Math.max(1, Math.round(img.width * scale));
          canvas.height = Math.max(1, Math.round(img.height * scale));
          canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', 0.85));
        };
        img.onerror = reject;
      });
    });
}

function extractRecognition(data) {
  const recognition = data?.recognition;
  if (!recognition || !recognition.calligrapher) return null;
  return { calligrapher: recognition.calligrapher, confidence: recognition.confidence };
}

async function exportConversationPdf() {
  if (state.exporting) return;
  const messages = state.messages;
  let selected = null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const user = messages[index];
    if (user.role !== 'user' || !user.previewUrl) continue;
    for (let next = index + 1; next < messages.length && messages[next].role !== 'user'; next += 1) {
      const ai = messages[next];
      if (ai.role === 'ai' && !ai.loading && ai.data) selected = { user, result: ai.data };
    }
    if (selected) break;
  }
  if (!selected) {
    alert('没有找到已完成鉴别的图片。');
    return;
  }
  state.exporting = true;
  const btn = $('exportPdfBtn');
  if (btn) btn.textContent = '导出中...';
  try {
    let image;
    try { image = await imageToDataUrl(selected.user.previewUrl, 1600); } catch {
      alert('最近一张图片无法读取，请重新上传后再试。');
      return;
    }
    const response = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image, result: selected.result }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.pdf?.data_url) {
      alert(data.message || `导出失败（HTTP ${response.status}）。`);
      return;
    }
    const link = document.createElement('a');
    link.href = data.pdf.data_url;
    link.download = data.pdf.filename || 'zhimo_report.pdf';
    link.click();
  } catch (err) {
    alert(`导出失败：${err.message || err}`);
  } finally {
    state.exporting = false;
    if (btn) btn.textContent = '导出为PDF';
  }
}

const exportPdfBtn = $('exportPdfBtn');
if (exportPdfBtn) exportPdfBtn.addEventListener('click', exportConversationPdf);

// ===== 欢迎页进入逻辑 =====
function enterApp() {
  const storage = zhimoStorage();
  if (storage) storage.setItem('zhimoEntered', '1');
  const welcome = $('welcome');
  if (welcome) welcome.classList.add('hidden');
}

const enterBtn = $('enterBtn');
if (enterBtn) enterBtn.addEventListener('click', enterApp);

// ===== 重新查看欢迎页 =====
function showWelcome() {
  const storage = zhimoStorage();
  if (storage) storage.removeItem('zhimoEntered');
  const welcome = $('welcome');
  if (welcome) welcome.classList.remove('hidden');
}

const welcomeBtn = $('welcomeBtn');
if (welcomeBtn) welcomeBtn.addEventListener('click', showWelcome);
