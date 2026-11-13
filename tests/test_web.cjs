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

test('library keeps distinct fixed revisions and removes only branch placeholders', () => {
  const page = app();
  page.run(`
    const base = {repo: 'org/model', variant: 'model.gguf', task: 'chat', jobs: []};
    state.catalog = {tasks: [{id: 'chat', label: 'Chat', group: '语言'}], candidates: [
      {...base, id: 'placeholder', commit_rev: 'main'},
      {...base, id: 'old', commit_rev: 'aaaa', seed: {id: 'old', offline_ready: true, integrity: 'ok'}},
      {...base, id: 'new', commit_rev: 'bbbb'},
    ]};
    renderLibrary();
  `);
  const html = page.elements.get('#librarySections').innerHTML;
  assert.match(html, /data-try="old"/);
  assert.match(html, /data-download="new"/);
  assert.doesNotMatch(html, /data-download="placeholder"/);
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
  assert.match(page.elements.get('#labTasks').innerHTML, /data-lab-task="chat" disabled/);
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
  assert.equal(page.run(`JSON.stringify(calls)`), '["/api/status","/api/catalog","/api/status"]');
});
