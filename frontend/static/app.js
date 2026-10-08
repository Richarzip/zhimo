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
    return `<div class="chat-msg ai"><div class="chat-avatar ai">${AI_ICON}</div><div class="chat-bubble">${body}</div></div>`;
  }).join('');
  scrollChat();
}

const AI_ICON = `<svg viewBox="0 0 1024 1024"><path d="M485.747988 62.049638C443.821802 66.47593701 421.319145 83.567421 421.319145 110.985477 421.319145 119.346659 423.411146 129.308423 426.114458 134.470669 427.590599 137.051814 427.22156401 137.667566 422.303954 141.724787 393.532074 166.192686 378.408064 203.570383 379.761909 247.22168L380.006502 257.179197 371.27624199 267.507894C353.817869 288.287585 338.20252699 315.830062 328.365161 343.003504 318.283203 371.037313 310.168717 412.71899201 313.118854 421.571548 316.071137 430.668697 325.78406 435.217207 333.651806 431.284471 340.414593 427.965299 343.366875 422.801036 344.349539 412.225514 346.194715 392.921965 352.46617 368.209473 360.70295201 347.674375L365.12923 336.23857201 370.909353 341.64965999C374.106228 344.60194201 379.02383799 348.290149 381.85167799 349.641848 389.105796 353.454498 401.032931 356.157896 407.795782 355.666564L413.573738 355.297529 419.967446 363.289718C427.590599 372.880344 437.672558 381.488307 447.387627 386.652656 451.320513 388.742425 454.517303 390.954492 454.517303 391.57035101 454.517303 393.291 445.295754 400.545118 429.927108 411.242765 413.573738 422.431915 402.38463 432.14475201 393.40977801 442.595832 373.735047 465.712031 345.703384 515.877295 340.047703 538.008621 336.973124 549.937901 336.481792 572.191588 338.940597 585.839393 342.628805 606.003374 352.83306 631.087047 364.391159 648.17853 371.400684 658.62737801 388.245429 676.456933 394.514652 680.269582 396.359915 681.374628 397.958352 682.850684 397.958352 683.711051 397.958352 685.187278 389.843867 748.755667 385.048554 784.41263 374.59755999 863.472006 374.59756 863.84104101 379.392873 873.184929 383.81915099 881.917334 390.457495 886.588205 404.474443 891.014483L417.261902 894.949429 413.820519 900.60511C409.76319099 907.367896 405.459209 919.786363 404.474443 927.77855201 402.875962 941.42427601 410.499202 953.229114 424.393767 958.271166 433.984265 961.837077 455.502112 963.557811 470.870673 961.961519 490.912336 959.991901 497.428319 956.42599 503.946577 944.254262 506.283085 939.827984 506.405382 937.98280699 505.791754 923.965902 505.420573 915.482382 504.80490601 907.612489 504.315527 906.383087 503.57754199 904.537911 504.315527 904.293317 511.078399 904.293317L518.823849 904.293317 517.963482 917.203116C516.978608 932.080388 518.332517 936.631108 524.726182 941.548718 534.68387 948.927278 563.333453 955.687919 589.152965 956.79502501 616.819885 958.024427 634.894032 953.844888 641.654608 944.745593 653.584018 928.63891901 646.943463 908.35056 625.548041 895.071726 619.40105 891.381373 618.662916 889.538342 623.089194 889.538342 627.884507 889.538342 639.689345 884.004958 644.606955 879.456384 650.507229 874.047442 653.704104 866.424288 653.704104 857.57173199 653.704104 854.497153 651.247445 831.258657 648.172866 805.928246 637.721872 718.016315 634.033729 687.279107 633.540187 684.08223199 632.926559 680.88535601 633.048855 680.88535601 640.794306 686.047645 661.698439 700.18676001 684.076567 694.286573 695.510225 671.786061 701.534876 659.981223 704.240484 645.103951 705.961219 614.120005 706.454696 606.127859 707.806395 596.659422 709.158094 592.110912 714.691478 573.543287 710.018461 555.224654 696.370506 542.559341L691.575279 538.010766 695.87926 529.649607C703.62471 514.523386 711.49453799 493.990434 711.494538 488.70378901 711.494538 481.818706 709.035798 476.532061 703.749153 471.861146 695.756921 464.851664 685.181528 463.991297 663.172434 468.53982899 658.501563 469.524681 654.319878 470.260692 653.70625 470.260692 653.090476 470.260692 648.295163 464.604926 643.008517 457.71984301 631.819453 442.842571 617.680166 427.718517 609.072289 421.324809 605.63082 418.866004 596.284787 412.594464 588.170237 407.554643 579.931438 402.512591 573.170733 397.717342 572.923951 397.10365 572.801698 396.365643 577.966132 393.291 584.604476 390.094124 598.868034 383.08674499 610.301756 374.10974699 617.924845 363.660899 620.999424 359.477068 624.07406701 356.280193 624.812073 356.404635 627.76221001 357.142705 638.951189 400.7919 638.951189 411.858625 638.951189 431.775803 660.96036801 438.907625 670.182024 422.062923 672.640764 417.636602 672.640764 417.636602 669.935221 401.283103 666.247099 378.536025 661.08272899 359.110178 654.44438501 342.265434L648.786494 328.00179 651.369741 321.977074C652.845883 318.780199 654.31987801 312.755483 654.688828 308.575944 655.426941 299.107614 651.738777 290.008319 644.853694 284.108045L640.058381 279.926361 641.532441 262.959319C645.714061 213.654487 636.737149 179.47152 613.129532 153.896516 603.16987 143.200929 592.22754401 134.839683 581.03848001 129.553059 573.170733 125.740453 572.432598 125.12678101 573.170733 122.298941 573.539768 120.576061 574.153483 114.675788 574.644814 109.142404 575.505182 96.230502 573.046291 88.731748 564.809509 79.63245301 557.677687 71.887003 547.718068 66.722654 534.43919101 64.141553 522.634353 61.68060199 498.659932 60.697939 485.747988 62.049638ZM531.489054 95.739127C534.808227 96.846233 538.987766 99.060445 540.46390699 100.658926 542.922798 103.239984 543.29174699 104.71398 542.800415 110.00277101 542.431444 113.444218 541.693309 116.7656 541.07753499 117.25691 540.463907 117.87262 534.439191 117.75040899 527.676405 117.01015001 511.691984 115.536155 483.658218 117.503628 468.287469 121.436407L456.853811 124.264311 455.008635 120.08470801C452.54983 113.93771701 452.796568 105.945528 455.624409 103.362281 459.19032 100.165405 470.010284 96.354944 480.952674 94.509725 493.74017601 92.419955 522.881092 93.035772 531.489054 95.739127ZM552.2666 152.298099C565.914469 156.479763 575.874131 161.644111 583.86632 168.898229 593.087911 177.137157 601.57357799 190.046956 605.015132 200.866985 608.45866 212.056049 609.688063 218.203062 608.45866 218.203062 607.965183 218.203062 603.538905 217.095956 598.621231 215.74425699 591.367177 213.776784 565.914469 209.597244 561.610488 209.597244 560.996859 209.597244 561.73493 207.25859001 563.086543 204.432896 566.65254 197.301074 567.268357 186.358748 564.56277 179.226927 554.478666 152.913809 517.594446 150.33275 504.684691 175.04526299 500.380602 183.284127 499.88939899 196.07165 503.577542 204.186157L505.91413601 209.350506 493.37114101 210.210873C468.534207 211.931607 444.804423 216.973659 423.411146 225.088145 417.261902 227.426798 412.221996 229.145387 411.975257 229.023091 411.361629 228.409462 415.049836 210.457611 417.139606 204.06386 425.869865 177.38175 446.156122 158.814125 474.189888 152.053485 489.191602 148.363132 489.560637 148.363132 516.609637 148.60987 539.356801 148.978927 542.06228 149.225644 552.268745 152.300223ZM534.43696 192.505761C534.43696 193.85746 533.947859 194.595487 533.209789 194.226495 532.594015 193.85746 531.98038601 192.628036 531.980386 191.645394 531.98038601 190.660584 532.594015 189.922514 533.209789 189.922514 533.947859 189.922514 534.439191 191.02962 534.439191 192.505761ZM548.578392 240.334452C570.218451 241.686151 585.342461 244.147101 599.35936599 248.573379L609.44347 251.77025499 609.44347 261.729917C609.44347 275.622379 606.12215199 310.541271 604.03238201 320.625375 597.391893 350.995693 570.711928 368.945398 528.167737 371.528645 514.028536 372.389012 491.52811 370.176946 480.33911 366.733332 455.502112 359.110178 438.77966399 342.879062 433.123898 320.625375 430.540693 310.912452 426.36119701 279.435029 425.992162 267.385597L425.623126 258.164006 435.829527 254.106763C453.659081 246.974941 478.740608 242.177483 507.38811 240.458894 529.521581 239.107195 528.414475 239.107195 548.578392 240.334452ZM397.960498 304.149666C398.696487 310.418975 399.92582599 317.181761 400.417157 319.026938 401.768856 323.57551199 400.91063501 324.191286 396.72895 322.099371 392.302586 319.76500801 388.736761 315.33873 386.15566 308.575944L384.063744 303.040414 389.352535 296.157477C393.0407 291.48445999 395.008216 289.886022 395.499548 290.993128 395.993025 291.85135 397.097985 297.755915 397.958352 304.149666ZM514.519867 402.265853C528.167737 402.63703299 539.356801 403.128365 539.356801 403.497401 539.356801 404.602361 533.578824 415.300093 525.462193 428.94796301L516.734079 443.580641 512.552351 438.785328C501.36333 425.875594 497.306087 420.955774 490.912336 411.981007 483.289183 401.40548501 482.673366 399.440158 486.857239 400.7919 488.331235 401.285334 500.749766 401.898963 514.519867 402.267998ZM468.905388 435.586371C471.731169 439.399042 480.461342 450.218986 488.206793 459.687316 496.076728 469.031204 502.47043599 476.901096 502.47043599 477.270131 502.47043599 477.514724 499.026822 487.35209 494.969579 498.910125 490.790104 510.590671 487.224129 520.672501 487.101832 521.410615 486.732797 522.395338 484.88762 522.395338 480.092307 521.288318 476.526439 520.427951 467.551544 518.336036 460.052832 516.493005 439.640031 511.697691 421.196848 509.36114 407.671276 509.852515 395.868583 510.221507 395.499548 510.345992 391.320008 514.40109 386.15566 519.320802 384.801815 524.729787 386.891585 531.123538 389.105796 537.641731 394.147848 540.345129 404.721139 540.469571 413.695991 540.469571 425.747569 541.943524 426.852529 542.92837601 427.221564 543.29741199 425.745466 546.863322 423.655653 550.673826 418.491305 560.142113 416.77057 566.164727 414.55850399 583.135973 413.451355 591.003806 411.237186 607.970912 409.516452 620.880646 407.795782 633.668062 406.072838 645.103951 405.703803 646.086615 405.336913 647.438314 403.244998 645.964318 398.696487 640.79997 376.195998 615.84074 363.408496 570.224115 371.400684 543.542005 378.163471 520.672501 409.147481 469.891528 425.747569 454.40067 433.370808 447.146617 459.681651 428.70337 462.509492 428.579013 463.247562 428.579013 466.075402 431.65342099 468.903243 435.588453ZM574.644814 436.695558C590.137774 446.161743 601.818171 456.243701 611.408797 468.293133 615.712779 473.70422 619.27869001 478.62183 619.27869 478.990823 619.27869 481.080635 600.835507 482.80355801 587.923563 482.187741 572.554959 481.327374 563.333453 479.359987 549.06978899 473.826453 541.446571 470.87638 539.356801 469.646978 539.972575 468.04860401 541.077535 465.2207 561.857226 429.686033 562.350704 429.686033 562.717594 429.686033 568.128724 432.88290901 574.644814 436.695558ZM674.606178 499.156928C674.606178 499.523732 670.795588 507.762745 666.124717 517.600111 658.992895 532.353026 657.147804 535.429665 654.935738 535.429665 650.509374 535.429665 638.582239 541.943524 633.171152 547.354654 622.475565 557.927945 616.328467 576.126534 617.064478 594.692013L617.433513 605.265304 598.990395 610.060617C583.25275601 613.995649 578.701971 614.733634 567.637349 614.733634 556.939617 614.61133699 533.578824 611.78345399 527.429602 609.81602399 526.324792 609.446988 527.060631 607.47958 529.521581 603.422273 542.06228 583.747585 538.249631 557.805648 521.282654 546.616584 518.20807501 544.526857 515.135684 542.681637 514.395425 542.437044 513.537204 542.190306 515.38238 535.54981599 519.315181 524.238455 522.758795 514.525575 525.955713 505.30394 526.447045 503.705546 526.816038 502.229361 527.676405 501.002104 528.167737 501.002104 528.661214 501.002104 533.94785901 502.720693 540.094872 504.688166 574.153483 516.370708 602.064909 517.353372 634.649439 508.131781 645.836443 504.934905 670.059663 499.034631 673.74787 498.66345099 674.239202 498.541154 674.606178 498.787893 674.606178 499.156928ZM469.888116 553.626023C470.626123 554.11735501 472.10011799 555.960472 473.329521 557.805648 475.788326 561.862977 480.092307 564.199399 490.665597 567.396275 501.607923 570.839889 504.929241 573.667729 504.929241 579.937038 504.929241 584.118723 504.19117 585.717204 499.764892 590.143439 495.709795 594.19845 493.002192 595.67682301 487.962199 596.781783 469.394575 600.716729 452.058498 595.061049 448.37029 583.994281 446.525114 578.3386 448.492587 569.732783 453.165604 562.231926 457.591882 555.224654 466.075402 550.796123 469.888116 553.626023ZM675.837597 568.378938C678.049706 570.101754 678.787777 572.313842 679.156812 577.355937 679.772522 584.856836 677.31155 590.881574 672.149432 594.692013 667.600858 597.888889 658.868453 598.749256 653.337215 596.53719 649.033233 594.692013 648.78864 594.322978 648.78864 588.914036 648.78864 582.026722 652.721441 572.313842 656.53409 569.977376 662.803399 565.922279 672.024926 565.306505 675.837597 568.378938ZM455.746619 627.029868C459.437058 628.259206 463.002969 629.610905 463.492241 630.102302 464.60140701 631.087047 459.312616 643.87454899 453.656936 653.587515 448.24792901 663.055759 442.345575 666.497271 437.427965 663.424923 436.076266 662.564556 436.198562 659.367595 438.410629 643.505471 441.854243 619.651244 441.360722 621.005088 445.54245 623.094944 447.509923 624.077564 452.05849801 625.798256 455.746619 627.029868ZM672.518468 634.406305C670.673227 652.11133101 667.107466 662.070993 662.558806 662.070993 657.147804 662.070993 648.419648 647.316017 646.943463 635.63781L646.083096 629.488609 673.0098 629.488609 672.516322 634.406305ZM515.013431 640.061899L527.060631 641.41359799 527.67640501 645.35069C528.905807 651.619999 527.307412 703.508078 525.95571299 706.213622 524.11049401 709.65509 524.357233 714.941736 526.447045 719.001124 527.554108 721.21319 527.798766 723.30296 527.18518 725.148093 525.2176 730.557078 523.372509 744.329389 521.529392 768.428253 518.946145 802.364481 519.070588 811.092595 522.020789 814.658506 524.357233 817.486346 524.970861 817.610788 541.815606 817.610788 558.90709 817.61078799 559.151618 817.610788 561.612569 814.536209 563.824657 811.832811 563.949142 810.232228 563.333453 799.167605 560.505527 749.862838 558.784793 733.384918 555.832511 726.255242 554.358515 722.689331 554.358515 721.582225 555.956996 718.632153 558.291316 714.203665 558.291316 707.934356 555.956996 703.87707 553.98948 700.433564 553.618299 685.309532 554.849847 661.210626L555.710214 646.333354 572.432598 645.59528299C581.654254 645.226334 591.120438 644.61261901 593.459092 644.119056 597.76092799 643.383217 597.883225 643.505471 598.129877 647.56275599 598.25225999 649.776968 599.237069 658.505082 600.219733 666.990662 601.204542 675.474269 604.89275 706.458301 608.211922 735.843723 611.655536 765.231377 616.081814 802.609074 618.04933 818.840191 620.01676 835.069161 621.737494 850.315469 621.982088 852.52968L622.351123 856.833662 609.932592 860.766462C582.145586 869.74346 558.046722 872.815894 512.92143 872.695743 470.132645 872.695743 443.821802 869.74346001 418.122269 861.873568 409.269713 859.17017 407.795782 858.307657 407.795782 855.973295 407.795782 854.497153 409.147481 842.447722 410.868151 829.291184 416.523832 786.5024 428.820002 694.655522 429.066741 694.53108 429.188973 694.408826 432.38591299 694.777776 436.318713 695.515889 449.352954 697.727955 463.494215 691.949978 473.576259 680.269582 479.476533 673.382354 486.486058 660.594809 491.52811 647.685053 495.463142 637.358501 495.583207 637.236205 499.27356 637.972065 501.36332999 638.341229 508.495152 639.201596 515.011199 640.061899ZM469.518952 904.171021C470.626123 905.153685 473.698556 917.08082 474.805576 924.334938L475.666072 929.74602501 470.870673 930.48195C464.845914 931.589056 450.337764 931.589056 442.959203 930.606392 438.90410601 929.990618 437.303523 929.254693 437.303523 927.778552 437.303523 924.950712 442.71461 913.883944 447.140888 907.2456L451.075834 901.71007 459.80609401 902.572583C464.723703 902.941618 469.027749 903.679689 469.518952 904.171021ZM579.564484 905.400423C582.76136 907.367896 591.736212 912.407803 599.481598 916.711784L613.498567 924.581676 606.246594 925.688782C597.145154 927.040481 573.290884 924.950712 559.76745699 921.507098 554.358515 920.155398 549.807795 918.803699 549.563202 918.556961 549.194166 918.310222 549.43876 914.744311 549.932237 910.687068L550.668162 903.186211 557.677687 902.69488C573.293115 901.712216 573.539768 901.712216 579.562338 905.398278Z" fill="#fff"/><path d="M467.182508 254.720392C463.49224101 255.458462 458.207591 261.97451 458.207591 265.662717 458.207591 267.263301 459.806094 270.335734 461.771378 272.54994601L465.337332 276.484892 481.690831 276.976224C495.585396 277.467556 498.659932 277.22081699 501.979104 275.37778601 509.971293 271.073805 509.971293 262.959319 502.101486 257.0569 499.026822 254.844834 496.692459 254.475798 484.271846 254.22906 476.4041 254.106763 468.658585 254.351356 467.182508 254.72253701Z" fill="#fff"/><path d="M567.75956 255.089427C558.291316 257.0569 553.620444 264.066425 557.061913 270.951508 559.64516 275.746821 564.56277 277.22081699 577.96613201 277.220817 586.20506 277.22081699 590.260071 276.607188 593.456839 275.008751 601.0801 271.196101 601.57357799 262.467987 594.317228 257.0569 590.384513 254.106763 577.103534 253.121954 567.75956 255.089427Z" fill="#fff"/><path d="M475.66607199 287.058182C471.48649 289.639284 466.813473 297.755915 466.813473 302.42678601 466.813473 307.09980299 471.608786 315.21428799 475.666072 317.797535 481.446152 321.485743 490.174266 320.625375 495.093978 315.952359 510.093589 301.566419 493.12654801 276.484892 475.66607199 287.058182Z" fill="#fff"/><path d="M567.512907 287.427218C562.103965 291.11757 558.415758 299.721242 559.522864 306.114993 560.014196 308.575944 562.226262 312.508744 564.440474 314.969695 571.941395 323.208622 585.342461 321.363446 590.50681 311.40378401 595.05547 302.551228 591.982887 291.362164 583.619581 286.935886 577.966132 283.86130699 572.554959 283.983603 567.512907 287.427218Z" fill="#fff"/><path d="M542.675887 340.666996C540.586204 342.756765 531.980386 343.494836 523.741459 342.140991 516.609637 340.911589 514.519867 341.036031 512.92143 342.38773 510.093589 344.724238 511.076253 349.76629 515.135684 352.716427 518.208075 354.92849401 520.422287 355.297529 531.364612 355.297529 543.169451 355.297529 544.274497 355.052936 547.226693 352.347392 551.161575 348.657039 551.281855 344.355203 547.718068 341.896398 544.767888 339.806629 543.783165 339.55989 542.675887 340.666996Z" fill="#fff"/></svg>`;

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
