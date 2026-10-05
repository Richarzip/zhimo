const state = {
  file: null,
  mode: 'single',
  previewUrl: null,
  fileVersion: 0,
  requestId: 0,
  controller: null,
  analyzing: false,
  validating: false,
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
  const canSend = Boolean(state.file) || $('prompt').value.trim();
  $('analyzeBtn').disabled = !canSend || state.analyzing || state.validating;
  $('analyzeBtn').textContent = state.analyzing ? '对话中...' : '发送';
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
  if (state.analyzing || state.validating) return;
  const text = $('prompt').value.trim();
  if (!state.file && !text) return;
  const requestId = ++state.requestId;
  const controller = new AbortController();
  state.controller = controller;
  state.analyzing = true;
  updateAnalyzeButton();
  $('runState').textContent = '运行中';
  renderTimeline([{ name: 'request', status: 'running', detail: '正在发送消息到本地服务' }]);

  // 用户消息入对话流 + AI loading 占位
  pushUserMessage(text, state.previewUrl);
  pushAiMessage({ loading: true });

  try {
    const form = new FormData();
    form.append('mode', 'chat');
    form.append('session_id', state.sessionId);
    form.append('text', text);
    if (state.file) form.append('image', state.file);
    form.append('rag', String($('ragToggle').checked));
    form.append('cam', String($('camToggle').checked));
    form.append('tta', String($('ttaToggle').checked));
    form.append('prompt', text);

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
      const message = data.message || `请求失败（HTTP ${res.status}）。`;
      renderResult({ ...data, message });
      pushAiMessage({ text: '', error: message });
      $('runState').textContent = '失败';
      return;
    }
    // 对话回复 + 右侧最新结果区（chat 返回无 summary，用 AI 回复代替）
    data.summary = data.reply || data.summary;
    pushAiMessage({ text: data.reply || '', data });
    renderResult(data);
    $('prompt').value = '';
    const partial = data.knowledge_diagnostic || data.recognition?.evidence?.error || data.steps?.some((step) => step.status === 'error');
    $('runState').textContent = data.blocked ? '已停止在可诊断节点' : partial ? '部分完成' : '完成';
  } catch (err) {
    if (requestId !== state.requestId || err.name === 'AbortError') return;
    const data = { error: 'browser_error', diagnostic: { message: '浏览器请求失败', raw: String(err) } };
    pushAiMessage({ text: '', error: '浏览器请求失败' });
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

// ====================== 多轮对话渲染 ======================

function pushUserMessage(text, previewUrl) {
  state.messages.push({ role: 'user', text: text || '', previewUrl: previewUrl || null });
  renderChat();
}

function pushAiMessage({ text = '', loading = false, data = null, error = '' } = {}) {
  state.messages.push({ role: 'ai', text, loading, data, error });
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
    return `<div class="chat-msg ai"><div class="chat-bubble">${body}</div></div>`;
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
  if (rec.evidence?.heatmap) {
    const src = rec.evidence.heatmap.startsWith('data:') ? rec.evidence.heatmap : `data:image/png;base64,${rec.evidence.heatmap}`;
    parts.push(`<figure class="chat-visual"><figcaption>Grad-CAM 热力图</figcaption><img src="${src}" alt="Grad-CAM 热力图"></figure>`);
  }
  if (data.inversion?.was_inverted) {
    parts.push('<div class="chat-note">检测到黑底白字拓印，已自动反转为白底黑字后再识别。</div>');
  }
  return parts.join('');
}

// 极简 Markdown 渲染（本地无外部依赖，覆盖标题/列表/引用/代码块/表格/粗体斜体）
function renderMarkdown(text) {
  const lines = String(text || '').split('\n');
  let html = '';
  let listTag = null; // 'ul' | 'ol' | null
  const closeList = () => { if (listTag) { html += `</${listTag}>`; listTag = null; } };
  const openList = (tag) => { if (listTag !== tag) { closeList(); html += `<${tag}>`; listTag = tag; } };
  let inCode = false;
  let codeBuf = [];
  let tableRows = null; // 表格行缓存（escape 后）
  const flushTable = () => {
    if (!tableRows || !tableRows.length) return;
    const rows = tableRows;
    tableRows = null;
    const sepRow = rows[0] && rows[0].every((c) => /^:?-+:?$/.test(c));
    const head = sepRow ? rows[1] : rows[0];
    const body = sepRow ? rows.slice(2) : rows.slice(1);
    if (!head || !body.length) return;
    const thead = `<thead><tr>${head.map((c) => `<th>${inlineMarkup(c)}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inlineMarkup(c)}</td>`).join('')}</tr>`).join('')}</tbody>`;
    html += `<table>${thead}${tbody}</table>`;
  };
  for (const raw of lines) {
    const trimmed = raw.trim();
    if (/^```/.test(trimmed)) {
      if (inCode) {
        html += `<pre><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`;
        codeBuf = [];
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeBuf.push(raw); continue; }
    const esc = escapeHtml(raw.trim());
    if (/^\|.*\|$/.test(esc)) {
      // 表格行：按管道拆单元格（已转义）
      tableRows = tableRows || [];
      tableRows.push(esc.split('|').slice(1, -1).map((c) => c.trim()));
      continue;
    }
    let m;
    if ((m = esc.match(/^####\s+(.*)$/))) { closeList(); flushTable(); html += `<h4>${m[1]}</h4>`; }
    else if ((m = esc.match(/^###\s+(.*)$/))) { closeList(); flushTable(); html += `<h3>${m[1]}</h3>`; }
    else if ((m = esc.match(/^##\s+(.*)$/))) { closeList(); flushTable(); html += `<h2>${m[1]}</h2>`; }
    else if ((m = esc.match(/^#\s+(.*)$/))) { closeList(); flushTable(); html += `<h1>${m[1]}</h1>`; }
    else if ((m = esc.match(/^&gt;\s?(.*)$/))) { closeList(); flushTable(); html += `<blockquote>${m[1]}</blockquote>`; }
    else if ((m = esc.match(/^[-*]\s+(.*)$/))) { flushTable(); openList('ul'); html += `<li>${m[1]}</li>`; }
    else if ((m = esc.match(/^\d+\.\s+(.*)$/))) { flushTable(); openList('ol'); html += `<li>${m[1]}</li>`; }
    else if (esc === '') { closeList(); flushTable(); }
    else {
      closeList(); flushTable();
      html += `<p>${inlineMarkup(esc)}</p>`;
    }
  }
  closeList();
  flushTable();
  if (inCode) html += `<pre><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`;
  return html;
}

function inlineMarkup(esc) {
  const bold = esc.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  return bold.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
}

function clearChat() {
  state.messages = [];
  $('chatList').innerHTML = CHAT_EMPTY_HTML;
}

// 新对话：重置会话记忆，保留已上传的图片与选项
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
  $('newChatBtn').addEventListener('click', resetChat);
  $('prompt').addEventListener('input', updateAnalyzeButton);
  renderTimeline([]);
  clearChat();
  loadHealth();
}

boot();
