const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');

function fixture({deferImages = false, healthStatus = 200} = {}) {
  const elements = new Map();
  for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) {
    const classes = new Set();
    const attributes = new Map();
    const listeners = new Map();
    elements.set(id, {
      textContent: '', innerHTML: '', value: '', src: '', disabled: false,
      checked: ['camToggle', 'ragToggle'].includes(id), listeners,
      classList: {
        add: (name) => classes.add(name), remove: (name) => classes.delete(name),
        contains: (name) => classes.has(name),
        toggle: (name, force) => force ? classes.add(name) : classes.delete(name),
      },
      addEventListener(name, callback) { listeners.set(name, callback); },
      setAttribute(name, value) { attributes.set(name, value); },
      getAttribute(name) { return attributes.get(name); },
      removeAttribute(name) { attributes.delete(name); if (name === 'src') this.src = ''; },
    });
  }
  const el = (id) => elements.get(id) || null;
  const pending = [];
  const images = [];
  const urls = new Map();
  const revoked = [];
  const alerts = [];
  const downloads = [];
  let nextUrl = 0;
  class MockImage {
    set src(url) {
      const file = urls.get(url);
      this.naturalWidth = file.width ?? 100;
      this.naturalHeight = file.height ?? 100;
      this.succeed = () => this.onload();
      this.fail = () => this.onerror();
      images.push(this);
      if (!deferImages) queueMicrotask(() => file.corrupt ? this.fail() : this.succeed());
    }
  }
  const modes = Object.fromEntries(['auto', 'single', 'multi'].map((mode) => {
    const listeners = new Map();
    return [mode, {
      dataset: {mode}, listeners,
      classList: {add() {}, remove() {}},
      addEventListener(name, callback) { listeners.set(name, callback); },
    }];
  }));
  const context = vm.createContext({
    alert: (message) => alerts.push(String(message)),
    document: {
      getElementById: el,
      querySelectorAll: () => Object.values(modes),
      createElement: (tag) => ({
        tagName: String(tag).toUpperCase(),
        href: '', download: '',
        click: () => downloads.push(String(tag)),
      }),
    },
    URL: {
      createObjectURL(file) { const url = `blob:${++nextUrl}:${file.name}`; urls.set(url, file); return url; },
      revokeObjectURL(url) { revoked.push(url); },
    },
    Image: MockImage, AbortController,
    FormData: class { constructor() { this.entries = new Map(); } append(key, value) { this.entries.set(key, value); } },
    fetch(url, options) {
      if (url === '/api/health') return Promise.resolve({ok: healthStatus === 200, status: healthStatus, json: async () => ({model_exists: true})});
      // Deliberately allow completion after abort: request identity must protect the UI too.
      return new Promise((resolve, reject) => pending.push({options, resolve, reject}));
    },
  });
  vm.runInContext(source + '\nthis.api = {setFile, analyze, resetAll, resetChat, renderQuality};', context);
  return {api: context.api, el, pending, images, revoked, modes, alerts, downloads};
}
const file = (name = 'image.png', extra = {}) => ({name, type: 'image/png', size: 1024, ...extra});
const success = (author = '王羲之') => ({mode: 'single', recognition: {calligrapher: author, confidence: 0.8}, quality: {overall: 0}});
function respond(request, data, status = 200) {
  request.resolve({ok: status >= 200 && status < 300, status, json: async () => data});
}

test('reset aborts a pending request and ignores a late result', async () => {
  const f = fixture();
  await f.api.setFile(file());
  const request = f.api.analyze();
  f.api.resetAll();
  assert.equal(f.pending[0].options.signal.aborted, true);
  respond(f.pending[0], success());
  await request;
  assert.equal(f.el('runState').textContent, '等待输入');
  assert.equal(f.el('calligrapher').textContent, '-');
  assert.equal(f.el('preview').src, '');
  assert.equal(f.el('resultMode').textContent, '未运行');
  assert.equal(f.el('analyzeBtn').disabled, true);
});

test('old response and finally cannot affect a new pending request', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const a = f.api.analyze();
  await f.api.setFile(file('B.png'));
  assert.equal(f.pending[0].options.signal.aborted, true);
  const b = f.api.analyze();
  respond(f.pending[0], success('A'));
  await a;
  assert.equal(f.el('analyzeBtn').disabled, true);
  assert.equal(f.el('runState').textContent, '运行中');
  assert.equal(f.el('calligrapher').textContent, '-');
  respond(f.pending[1], success('B'));
  await b;
  assert.equal(f.el('calligrapher').textContent, 'B');
  // 图片已随发送清空输入栏预览（对话气泡中的图片不受影响）
  assert.equal(f.el('preview').src, '');
});

test('old failure cannot overwrite a newer completed result', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const a = f.api.analyze();
  await f.api.setFile(file('B.png'));
  const b = f.api.analyze();
  respond(f.pending[1], success('B'));
  await b;
  f.pending[0].reject(new Error('late network failure'));
  await a;
  assert.equal(f.el('calligrapher').textContent, 'B');
  assert.equal(f.el('runState').textContent, '完成');
});

test('repeated Analyze clicks submit only one request', async () => {
  const f = fixture();
  await f.api.setFile(file());
  const a = f.api.analyze();
  await f.api.analyze();
  assert.equal(f.pending.length, 1);
  respond(f.pending[0], success());
  await a;
});

test('HTTP and business errors are failures even with valid JSON', async (t) => {
  for (const status of [200, 400, 413, 500, 503]) {
    await t.test(String(status), async () => {
      const f = fixture();
      await f.api.setFile(file());
      const p = f.api.analyze();
      respond(f.pending[0], {error: 'test_error', message: '请求失败'}, status);
      await p;
      assert.equal(f.el('runState').textContent, '失败');
      // 从未有过识别数据时，纯错误轮次保持结果区为空（不渲染"未评估"占位）
      assert.equal(f.el('qualityGrid').innerHTML, '');
      assert.doesNotMatch(f.el('qualityGrid').innerHTML, /0\.000/);
      // 图片已随发送清空；重新上传后按钮恢复可点（setFile 会清空结果区，放最后）
      await f.api.setFile(file());
      assert.equal(f.el('analyzeBtn').disabled, false);
    });
  }
});

test('HTTP failure is detected even without a business error field', async () => {
  const f = fixture();
  await f.api.setFile(file());
  const p = f.api.analyze();
  respond(f.pending[0], {message: '服务暂不可用'}, 503);
  await p;
  assert.equal(f.el('runState').textContent, '失败');
});

test('non-JSON HTTP errors retain the HTTP status in the diagnostic', async () => {
  const f = fixture();
  await f.api.setFile(file());
  const p = f.api.analyze();
  f.pending[0].resolve({ok: false, status: 502, json: async () => {throw new SyntaxError('HTML');}});
  await p;
  assert.equal(f.el('runState').textContent, '失败');
  assert.match(f.el('diagnostic').innerHTML, /HTTP 502/);
});

test('network failure and malformed JSON payloads recover the submit button', async (t) => {
  for (const payload of [null, [], 'invalid']) {
    await t.test(JSON.stringify(payload), async () => {
      const f = fixture();
      await f.api.setFile(file());
      const p = f.api.analyze();
      respond(f.pending[0], payload);
      await p;
      assert.equal(f.el('runState').textContent, '失败');
      await f.api.setFile(file());
      assert.equal(f.el('analyzeBtn').disabled, false);
    });
  }
  const f = fixture();
  await f.api.setFile(file());
  const p = f.api.analyze();
  f.pending[0].reject(new TypeError('offline'));
  await p;
  assert.equal(f.el('runState').textContent, '失败');
  await f.api.setFile(file());
  assert.equal(f.el('analyzeBtn').disabled, false);
});

test('blocked and partial results have distinct states', async (t) => {
  for (const [data, state] of [
    [{blocked: true, diagnostic: {message: '缺少权重'}}, '已停止在可诊断节点'],
    [{...success(), knowledge_diagnostic: {message: '检索失败'}}, '部分完成'],
    [{...success(), steps: [{status: 'error', name: 'search'}]}, '部分完成'],
    [success(), '完成'],
  ]) {
    await t.test(state, async () => {
      const f = fixture();
      await f.api.setFile(file());
      const p = f.api.analyze();
      respond(f.pending[0], data);
      await p;
      assert.equal(f.el('runState').textContent, state);
    });
  }
});

test('missing, null, and nonfinite quality differ from a genuine zero', () => {
  const f = fixture();
  f.api.renderQuality({yellow: 0, fade: null, noise: NaN, blur: Infinity});
  const output = f.el('qualityGrid').innerHTML;
  assert.equal((output.match(/未评估/g) || []).length, 4);
  assert.equal((output.match(/0\.000/g) || []).length, 1);
  assert.doesNotMatch(output, /NaN|Infinity/);
});

test('invalid uploads never enable Analyze and show a useful error', async (t) => {
  for (const [upload, message] of [
    [file('notes.txt', {type: 'text/plain'}), /JPEG/],
    [file('image.svg', {type: 'image/svg+xml'}), /JPEG/],
    [file('empty.png', {size: 0}), /为空/],
    [file('large.png', {size: 32 * 1024 * 1024 + 1}), /32 MiB/],
    [file('corrupt.png', {corrupt: true}), /无法解码/],
    [file('huge.png', {width: 5001, height: 5000}), /2500 万像素/],
  ]) {
    await t.test(upload.name, async () => {
      const f = fixture();
      await f.api.setFile(upload);
      await f.api.analyze();
      assert.equal(f.el('analyzeBtn').disabled, true);
      assert.match(f.el('uploadError').textContent, message);
      assert.equal(f.el('imageInput').getAttribute('aria-invalid'), 'true');
      assert.equal(f.pending.length, 0);
      assert.equal(f.el('preview').src, '');
    });
  }
});

test('the drop handler applies validation to non-image files', async () => {
  const f = fixture();
  await f.el('dropzone').listeners.get('drop')({dataTransfer: {files: [file('notes.txt', {type: 'text/plain'})]}});
  assert.equal(f.el('analyzeBtn').disabled, true);
  assert.match(f.el('uploadError').textContent, /JPEG/);
});

test('boundary-size images and missing MIME with a supported extension are accepted', async () => {
  const f = fixture();
  await f.api.setFile(file('photo.PNG', {type: '', size: 32 * 1024 * 1024, width: 5000, height: 5000}));
  assert.equal(f.el('analyzeBtn').disabled, false);
  assert.equal(f.el('uploadError').textContent, '');
});

test('switching files clears results and keeps history image URLs until reset', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const oldUrl = f.el('preview').src;
  const p = f.api.analyze();
  respond(f.pending[0], success('A'));
  await p;
  await f.api.setFile(file('B.png'));
  const newUrl = f.el('preview').src;
  assert.equal(f.el('calligrapher').textContent, '-');
  assert.equal(f.el('resultMode').textContent, '未运行');
  assert.deepEqual(f.revoked, []);
  f.api.resetAll();
  assert.deepEqual(new Set(f.revoked), new Set([oldUrl, newUrl]));
  assert.equal(f.revoked.length, 2);
});

test('out-of-order image validation cannot replace a newer file', async () => {
  const f = fixture({deferImages: true});
  const a = f.api.setFile(file('A.png'));
  assert.equal(f.el('analyzeBtn').disabled, true);
  const b = f.api.setFile(file('B.png'));
  f.images[1].succeed();
  await b;
  f.images[0].succeed();
  await a;
  assert.match(f.el('preview').src, /B\.png/);
  assert.equal(f.el('analyzeBtn').disabled, false);
  assert.equal(f.revoked.length, 1);
  assert.match(f.revoked[0], /A\.png/);
});

test('reset during image validation prevents late preview and releases its URL', async () => {
  const f = fixture({deferImages: true});
  const p = f.api.setFile(file());
  f.api.resetAll();
  f.images[0].succeed();
  await p;
  assert.equal(f.el('preview').src, '');
  assert.equal(f.el('analyzeBtn').disabled, true);
  assert.equal(f.el('runState').textContent, '等待输入');
  assert.equal(f.revoked.length, 1);
});

test('health HTTP failure never appears ready', async () => {
  const f = fixture({healthStatus: 500});
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.el('serverStatus').textContent, '服务异常');
});


test('each completed or failed request replaces its loading bubble', async (t) => {
  for (const outcome of ['success', 'business-error', 'network-error', 'abort']) {
    await t.test(outcome, async () => {
      const f = fixture();
      f.el('prompt').value = '介绍王羲之';
      const request = f.api.analyze();
      assert.equal((f.el('chatList').innerHTML.match(/chat-loading/g) || []).length, 1);
      if (outcome === 'success') respond(f.pending[0], {mode: 'chat', reply: '本轮回答'});
      else if (outcome === 'business-error') respond(f.pending[0], {error: 'test', message: '服务失败'}, 500);
      else if (outcome === 'network-error') f.pending[0].reject(new Error('offline'));
      else f.pending[0].reject(Object.assign(new Error('aborted'), {name: 'AbortError'}));
      await request;
      const chat = f.el('chatList').innerHTML;
      assert.doesNotMatch(chat, /chat-loading/);
      assert.equal((chat.match(/chat-msg ai/g) || []).length, 1);
      assert.match(chat, outcome === 'success' ? /本轮回答/ : /失败|取消/);
      // 发送后输入区（文本+图片）即被清空：请求结束后按钮应为禁用，重新输入会恢复
      assert.equal(f.el('analyzeBtn').disabled, true);
    });
  }
});

test('changing files finishes the cancelled bubble without affecting a newer reply', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const first = f.api.analyze();
  await f.api.setFile(file('B.png'));
  assert.doesNotMatch(f.el('chatList').innerHTML, /chat-loading/);
  assert.match(f.el('chatList').innerHTML, /取消/);
  const second = f.api.analyze();
  respond(f.pending[0], {mode: 'chat', reply: '过期回答'});
  await first;
  assert.equal((f.el('chatList').innerHTML.match(/chat-loading/g) || []).length, 1);
  assert.doesNotMatch(f.el('chatList').innerHTML, /过期回答/);
  respond(f.pending[1], {mode: 'chat', reply: '最新回答'});
  await second;
  assert.doesNotMatch(f.el('chatList').innerHTML, /chat-loading/);
  assert.match(f.el('chatList').innerHTML, /最新回答/);
  assert.equal((f.el('chatList').innerHTML.match(/chat-msg ai/g) || []).length, 2);
});

test('a later conversational turn keeps the latest recognition results', async (t) => {
  for (const secondResponse of [{mode: 'chat', reply: '普通回答'}, {error: 'test', message: '本轮失败'}]) {
    await t.test(secondResponse.error ? 'failure' : 'success', async () => {
      const f = fixture();
      await f.api.setFile(file());
      const first = f.api.analyze();
      respond(f.pending[0], {
        ...success(), reply: '首次回答',
        recognition: {calligrapher: '王羲之', evidence: {heatmap: 'data:image/png;base64,old'}},
        segmentation: {overlay: 'data:image/png;base64,boxes'},
        diagnostic: {message: '旧诊断'},
        reference_examples: [{url: '/old.jpg', title: '旧资料'}],
        reference_examples_note: '旧备注',
        pdf: {data_url: 'data:application/pdf;base64,old', filename: 'old.pdf'},
      });
      await first;
      for (const id of ['segFigure', 'diagnostic', 'referenceBox', 'reportActions']) {
        assert.equal(f.el(id).classList.contains('hidden'), false);
      }
      assert.equal(typeof f.el('downloadPdfBtn').onclick, 'function');
      f.el('prompt').value = '再问一次';
      const second = f.api.analyze();
      respond(f.pending[1], secondResponse, secondResponse.error ? 500 : 200);
      await second;
      // 纯对话/错误轮次不带识别产物：结果区保持上一轮最新非空数据
      for (const id of ['segFigure', 'diagnostic', 'referenceBox', 'reportActions']) {
        assert.equal(f.el(id).classList.contains('hidden'), false);
      }
      assert.equal(f.el('segImage').src, 'data:image/png;base64,boxes');
      assert.match(f.el('diagnostic').innerHTML, /旧诊断/);
      assert.match(f.el('referenceGrid').innerHTML, /old\.jpg/);
      assert.equal(f.el('referenceNote').textContent, '旧备注');
      assert.equal(typeof f.el('downloadPdfBtn').onclick, 'function');
      // 本轮信息仍更新：模式标签与原始 JSON
      assert.equal(f.el('resultMode').textContent, secondResponse.error ? '异常' : '对话');
      assert.match(f.el('rawJson').textContent, secondResponse.error ? /本轮失败/ : /普通回答/);
    });
  }
});

test('history retains image URLs across rerenders and a new chat releases only unused images', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const firstUrl = f.el('preview').src;
  // 第一轮：带图发送（发送后输入栏预览清空，但对话气泡保留图片引用）
  const request0 = f.api.analyze();
  respond(f.pending[0], {mode: 'chat', reply: '回答'});
  await request0;
  assert.equal(f.el('preview').src, '');
  // 第二轮：纯文本追问，不再自动携带旧图
  f.el('prompt').value = '追问';
  const request1 = f.api.analyze();
  respond(f.pending[1], {mode: 'chat', reply: '回答2'});
  await request1;
  // 重新上传 B 再分析
  await f.api.setFile(file('B.png'));
  const currentUrl = f.el('preview').src;
  const next = f.api.analyze();
  respond(f.pending[2], {mode: 'chat', reply: '新图片回答'});
  await next;
  // A、B 均被对话消息引用，未提前释放
  assert.deepEqual(f.revoked, []);
  assert.equal(f.el('chatList').innerHTML.split(firstUrl).length - 1, 1);
  f.api.resetChat();
  // resetChat 清空消息后释放全部图片（输入栏预览已空，无保留项）
  assert.deepEqual(f.revoked, [firstUrl, currentUrl]);
  assert.equal(f.el('preview').src, '');
  assert.doesNotMatch(f.el('chatList').innerHTML, /chat-img/);
  f.api.resetAll();
  assert.deepEqual(f.revoked, [firstUrl, currentUrl]);
  f.api.resetAll();
  assert.equal(f.revoked.length, 2);
});

test('unused image previews are released immediately when replaced', async () => {
  const f = fixture();
  await f.api.setFile(file('A.png'));
  const firstUrl = f.el('preview').src;
  await f.api.setFile(file('B.png'));
  assert.deepEqual(f.revoked, [firstUrl]);
});

test('reset starts a new server session and an old response cannot restore cleared history', async () => {
  const f = fixture();
  await f.api.setFile(file());
  f.el('prompt').value = '旧会话';
  const first = f.api.analyze();
  const oldSession = f.pending[0].options.body.entries.get('session_id');
  f.api.resetAll();
  assert.equal(f.pending[0].options.signal.aborted, true);
  assert.doesNotMatch(f.el('chatList').innerHTML, /旧会话|chat-loading/);
  f.el('prompt').value = '新会话';
  const second = f.api.analyze();
  assert.notEqual(f.pending[1].options.body.entries.get('session_id'), oldSession);
  respond(f.pending[1], {mode: 'chat', reply: '新回答'});
  await second;
  respond(f.pending[0], {mode: 'chat', reply: '旧回答'});
  await first;
  assert.match(f.el('chatList').innerHTML, /新会话|新回答/);
  assert.doesNotMatch(f.el('chatList').innerHTML, /旧会话|旧回答|chat-loading/);
});

test('analysis mode selection is sent while preserving conversational requests', async () => {
  const f = fixture();
  for (const [index, mode] of ['auto', 'multi', 'single'].entries()) {
    if (mode !== 'auto') f.modes[mode].listeners.get('click')();
    // 图片随发送清空，每次分析前重新上传
    await f.api.setFile(file());
    const request = f.api.analyze();
    const fields = f.pending[index].options.body.entries;
    assert.equal(fields.get('mode'), 'chat');
    assert.equal(fields.get('analysis_mode'), mode);
    assert.equal(f.el('camToggle').disabled, mode === 'multi');
    assert.equal(fields.get('cam'), mode === 'multi' ? 'false' : 'true');
    respond(f.pending[index], {mode: 'chat', reply: '回答'});
    await request;
  }
  assert.match(html, /class="active" data-mode="auto"/);
});


test('leaving multi-character mode restores the previous CAM preference', async (t) => {
  for (const checked of [false, true]) {
    await t.test(String(checked), () => {
      const f = fixture();
      f.el('camToggle').checked = checked;
      f.modes.multi.listeners.get('click')();
      f.modes.multi.listeners.get('click')();
      assert.equal(f.el('camToggle').disabled, true);
      assert.equal(f.el('camToggle').checked, false);
      f.modes.auto.listeners.get('click')();
      assert.equal(f.el('camToggle').disabled, false);
      assert.equal(f.el('camToggle').checked, checked);
    });
  }
});

test('agent export_pdf tool call triggers a conversation export request', async () => {
  const f = fixture();
  await f.api.setFile(file());
  f.el('prompt').value = '帮我导出成 PDF';
  const p = f.api.analyze();
  // Agent 回复并调用了 export_pdf 工具
  respond(f.pending[0], {
    mode: 'chat',
    reply: '好的，正在为您导出对话为 PDF。',
    steps: [{name: 'export_pdf', status: 'ok'}],
  });
  await p;
  // exportConversationPdf 内部先 fetch 用户图片（blob URL）→ 拒绝让它走 image=null 分支
  assert.ok(f.pending[1], '图片 fetch 应已发起');
  f.pending[1].reject(new Error('no blob in test'));
  await new Promise((resolve) => setTimeout(resolve, 0));
  // 随后应发起 /api/export 请求，携带对话 entries
  assert.ok(f.pending[2], '导出请求应已发起');
  const exportRequest = f.pending[2];
  assert.equal(exportRequest.options.method, 'POST');
  const body = JSON.parse(exportRequest.options.body);
  assert.ok(Array.isArray(body.entries));
  assert.equal(body.entries[0].role, 'user');
  assert.equal(body.entries[1].role, 'ai');
  respond(exportRequest, {pdf: {filename: 'zhimo_chat.pdf', data_url: 'data:application/pdf;base64,QUJD'}});
  await new Promise((resolve) => setTimeout(resolve, 0));
});
