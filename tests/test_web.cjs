const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/doomsday_seed/web/assets/app.js'), 'utf8');

function app() {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, { innerHTML: '', textContent: '', dataset: {} });
      return elements.get(selector);
    },
    querySelectorAll() { return []; },
  };
  const context = vm.createContext({ document, URL, setTimeout: () => 0, clearTimeout() {} });
  vm.runInContext(source.slice(0, source.indexOf("document.addEventListener('click'")), context);
  return { run: code => vm.runInContext(code, context), elements, context };
}

test('cave travel swaps rooms under black and retreats through the same gate', async () => {
  const page = app();
  page.run(`
    const location = { hash: '#/vault' };
    let reduce = false;
    const matchMedia = () => ({ matches: reduce });
    const moves = [];
    function animate(frames, timing) {
      let finish, abort;
      const finished = new Promise((ok, fail) => { finish = ok; abort = fail; });
      finished.catch(() => {});
      const motion = { finished, finish, cancel() { abort({ name: 'AbortError' }); } };
      moves.push({ frames, timing, motion });
      return motion;
    }
    const rock = { animate };
    Object.assign($('.cave-scene'), { animate, querySelector: () => rock });
    $('.cave-blackout').animate = animate;
    for (const view of ['library', 'lab', 'vault']) {
      $('#view-' + view).querySelector = () => ({ focus() {} });
      $('.gate[data-view="' + view + '"]').focus = () => {};
    }
    go('library');
    const entering = showView();
    showView();
  `);
  assert.equal(page.run('moves.length'), 3);
  assert.equal(page.run('state.view'), 'vault');
  assert.equal(page.run("$('.cave-scene').dataset.mouth"), 'library');
  await page.run('moves[2].motion.finish()');
  assert.equal(page.run('state.view'), 'library');
  assert.equal(page.run('moves[3].frames[0].opacity'), 1);
  await page.run('moves[3].motion.finish(); entering');
  page.run("go('vault'); const leaving = showView()");
  await page.run('moves[4].motion.finish()');
  assert.equal(page.run("$('.cave-scene').dataset.mouth"), 'library');
  assert.equal(page.run('moves[6].timing.direction'), 'reverse');
  await page.run('moves.slice(5).forEach(m => m.motion.finish()); leaving');
  assert.equal(page.run('state.view'), 'vault');
  assert.equal(page.run("$('#main').inert"), false);
  page.run("go('lab'); const interrupted = showView(); go('vault'); showView()");
  await page.run('interrupted');
  assert.equal(page.run('state.view'), 'vault');
  const count = page.run('moves.length');
  await page.run("go('library'); showView(false)");
  await page.run("reduce = true; go('lab'); showView()");
  assert.equal(page.run('state.view'), 'lab');
  assert.equal(page.run('moves.length'), count);
});

test('library shows a new recommendation alongside a previously archived revision', () => {
  const page = app();
  page.run(`
    const base = {repo: 'org/model', variant: 'model.gguf', task: 'chat', jobs: []};
    state.catalog = {tasks: [{id: 'chat', label: 'Chat', group: '语言'}], candidates: [
      {...base, id: 'old', commit_rev: 'aaaa', seed: {id: 'old', offline_ready: true, integrity: 'ok'}},
      {...base, id: 'new', commit_rev: 'bbbb', recommended_modes: ['balanced', 'accuracy']},
    ]};
    renderLibrary();
  `);
  const html = page.elements.get('#librarySections').innerHTML;
  assert.doesNotMatch(html, /data-try="old"/);
  page.run("state.libraryStage = 'stored'; renderLibrary()");
  assert.match(page.elements.get('#librarySections').innerHTML, /data-try="old"/);
  assert.match(html, /data-select-model="new"/);
  assert.equal((html.match(/data-select-model=/g) || []).length, 1);
  assert.match(html, /模型说明与接入/);
});

test('collected models expose validation and waiting downloads expose retry', () => {
  const page = app();
  assert.match(page.run(`modelActions({id: 'c', seed: {}}, 'stored')`), /验证本机试用/);
  assert.match(page.run(`modelActions({id: 'c', jobs: [{state: 'waiting_network', mode: 'sota'}]}, 'pending')`), /data-dmode="sota"/);
});

test('inference disables task changes while a request is running', () => {
  const page = app();
  page.run(`
    state.catalog = {tasks: [{id: 'chat', label: 'Chat', group: '语言', enabled: true}], defaults: [], candidates: []};
    state.status = {available_now: [{task: 'chat', seed_id: 'seed', variant: 'v'}]};
    state.chat = {};
    renderLab();
  `);
  assert.equal(page.elements.get('#labTasks').disabled, true);
  assert.equal(page.elements.get('#labModel').disabled, true);
});

test('trial type filter includes archive-only tasks and narrows the model list', () => {
  const page = app();
  page.run(`
    state.catalog = {tasks: [
      {id:'chat', label:'通用对话', group:'语言', enabled:true},
      {id:'tts', label:'语音合成', group:'音频', enabled:false}
    ], defaults:[], candidates:[]};
    state.status = {available_now:[{task:'chat', seed_id:'local', variant:'Chat.gguf'}]};
    renderLab();
  `);
  assert.match(page.elements.get('#labTasks').innerHTML, /全部类型.*音频.*语音合成/s);
  assert.equal(page.elements.get('#chatPanel').hidden, false);
  assert.equal(page.elements.get('#labModel').value, 'local');
  page.run("$('#chatLog').replaceChildren = () => {}; state.labFilter = 'tts'; renderLab()");
  assert.equal(page.elements.get('#labModel').disabled, true);
  assert.equal(page.elements.get('#labEmpty').hidden, false);
  assert.match(page.elements.get('#labEmptyText').textContent, /语音合成尚未接入本机试用/);
});

test('saving preferences refreshes the settings snapshot used by other forms', async () => {
  const page = app();
  await page.run(`
    state.settings = {subscriptions: {chat: ['balanced']}};
    state.preferences = {};
    api = async path => ({subscriptions: {chat: ['speed']}});
    renderSettings = renderLibrary = toast = () => {};
    savePreferences();
  `);
  assert.equal(page.run(`settingsPayload().subscriptions.chat[0]`), 'speed');
});

test('polling fetches only data that is used', async () => {
  const page = app();
  await page.run(`
    const calls = [];
    api = async path => { calls.push(path); return path === '/api/status' ? {network: {}, catalog_revision: '1'} : {}; };
    renderHome = renderLibrary = renderLab = () => {};
    (async () => { await refresh(); await refresh(); })();
  `);
  assert.equal(page.run(`JSON.stringify(calls)`), '["/api/status","/api/catalog?curated=true","/api/status"]');
});

test('overview distinguishes an unconnected archive from an empty one', () => {
  const page = app();
  page.run('renderHome()');
  assert.equal(page.elements.get('#homeReady').textContent, '—');
  page.run('state.status = { counts: { seeds: 4, offline_ready: 0, active_jobs: 2 } }; renderHome()');
  const counts = ['#homeStored', '#homeReady', '#homeJobs'].map(id => String(page.elements.get(id).textContent));
  assert.deepEqual(counts, ['4', '0', '2']);
});

test('library includes types without a recommendation and shows all four slots', () => {
  const page = app();
  page.run(`
    state.catalog = {tasks: [{id: 'voice', label: '语音'}], candidates: []};
    $('#libraryTask').value = 'voice';
    renderLibrary();
  `);
  assert.match(page.elements.get('#libraryTask').innerHTML, /语音/);
  const slots = page.elements.get('#librarySections').innerHTML;
  assert.equal((slots.match(/暂无合格版本/g) || []).length, 3);
  assert.match(slots, /暂无可比较的公开成绩/);
});


test('research summaries open in the details dialog and stay out of bulk downloads', () => {
  const page = app();
  page.run(`
    state.catalog = {tasks: [{id: 'image_edit', label: '图片编辑'}], candidates: [
      {id: 'research-image_edit', repo: 'org/research', variant: 'Research', task: 'image_edit',
       reference_only: true, recommended_modes: ['sota'], brief: {
        ability: '按指令编辑图像', effect: '公开榜单有结果', limit: '许可见发布页',
       }},
    ]};
    renderLibrary();
  `);
  const html = page.elements.get('#librarySections').innerHTML;
  assert.doesNotMatch(html, /按指令编辑图像|公开榜单有结果|许可见发布页|org\/research/);
  const details = page.run('modelGuide(state.catalog.candidates[0])');
  assert.match(details, /按指令编辑图像/);
  assert.match(details, /公开榜单有结果/);
  assert.match(details, /许可见发布页/);
  assert.match(html, /aria-haspopup="dialog"/);
  assert.doesNotMatch(html, /data-select-model/);
  assert.equal(page.run('state.visibleCandidates.length'), 0);
});
