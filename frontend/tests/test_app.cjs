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
  const context = vm.createContext({
    document: { getElementById: el, querySelectorAll: () => [] },
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
  vm.runInContext(source + '\nthis.api = {setFile, analyze, resetAll, renderQuality};', context);
  return {api: context.api, el, pending, images, revoked};
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
  assert.match(f.el('preview').src, /B\.png/);
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
      assert.equal(f.el('analyzeBtn').disabled, false);
      assert.match(f.el('qualityGrid').innerHTML, /未评估/);
      assert.doesNotMatch(f.el('qualityGrid').innerHTML, /0\.000/);
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
  assert.equal(f.el('summary').textContent, '服务暂不可用');
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
      assert.equal(f.el('analyzeBtn').disabled, false);
    });
  }
  const f = fixture();
  await f.api.setFile(file());
  const p = f.api.analyze();
  f.pending[0].reject(new TypeError('offline'));
  await p;
  assert.equal(f.el('runState').textContent, '失败');
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

test('switching files clears old results and releases old preview URLs', async () => {
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
  assert.deepEqual(f.revoked, [oldUrl]);
  f.api.resetAll();
  assert.deepEqual(f.revoked, [oldUrl, newUrl]);
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
