'use strict';

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[c]);
const say = (id, value) => { const el = $('#' + id); if (el && el.textContent !== String(value)) el.textContent = value; };
const state = {
  session: '', catalog: null, status: null, preferences: null,
  settings: null, schedule: null, agentConfig: null,
  labTask: null, model: '', view: 'vault', refresh: null,
  sync: false, pending: new Set(), toast: 0, chat: null, messages: [],
  image: null, fileVersion: 0, mediaRunning: false, initialized: false,
  prefDirty: false,
};
const labels = {
  queued: '排队中', downloading: '下载中', verifying: '校验中',
  debugging: '准备环境', retesting: '本机验证中', waiting_network: '等待来源',
  waiting_resource: '等待本机资源', failed: '准备未完成',
};
const profileLabels = { speed: '速度优先', balanced: '平衡', accuracy: '精度优先' };
const taskCopy = {
  pose: { description: '识别本地图片中的人体关键点，输出坐标与预览叠层', trial: '选择一张本地图片，输出人体关键点与预览叠层' },
  hand: { description: '识别本地图片中的手部关节，输出坐标与预览叠层', trial: '选择一张本地图片，输出每只手的关节坐标与预览叠层' },
};
const taskInfo = id => state.catalog?.tasks.find(t => t.id === id);
const taskField = (task, field) => task?.[field] || taskCopy[task?.id]?.[field] || '';
const ready = c => Boolean(c.seed?.offline_ready && c.seed.integrity === 'ok');
const waiting = job => job?.state.startsWith('waiting_');
const working = job => job && !['done', 'failed'].includes(job.state);
const activeJob = c => (c.jobs || []).find(job => job.state !== 'done') || null;
const size = n => n > 0 ? (n >= 1024**3 ? (n / 1024**3).toFixed(1) + ' GiB' : (n / 1024**2).toFixed(1) + ' MiB') : '';
const mib = n => n >= 1024 ? (n / 1024).toFixed(1) + ' GiB' : Math.round(n) + ' MiB';
const pct = n => Math.max(0, Math.min(100, Math.round((Number(n) || 0) * 100)));
const HF_BASE = 'https:' + '//huggingface.co';
const rendered = new WeakMap();
function render(id, value) {
  const el = $('#' + id);
  if (rendered.get(el) !== value) {
    el.innerHTML = value;
    rendered.set(el, value);
  }
}

function preferenceFor(task) {
  return state.preferences?.[task] || { mode: 'off', profiles: ['balanced'], sota_enabled: false };
}

function profilesFor(task) {
  const profiles = preferenceFor(task).profiles || [];
  if (!profiles.length) return ['balanced'];
  return profiles.includes('balanced')
    ? ['balanced', ...profiles.filter(p => p !== 'balanced')]
    : profiles;
}

async function api(path, method = 'GET', body, signal, stream = false) {
  const headers = { Accept: stream ? 'text/event-stream' : 'application/json' };
  if (method !== 'GET') headers['X-DS-Session'] = state.session;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    const d = data?.detail;
    throw new Error((typeof d === 'string' ? d : d?.error?.message) || '当前操作未完成，请稍后再试');
  }
  return stream ? res : res.json();
}

function toast(message) {
  say('toast', message);
  $('#toast').hidden = false;
  clearTimeout(state.toast);
  state.toast = setTimeout(() => { $('#toast').hidden = true; }, 4500);
}

function showView() {
  if (location.hash && !location.hash.startsWith('#/')) return;
  const requested = location.hash.slice(2);
  const alias = { discover: 'library', agent: 'settings' };
  const views = ['vault', 'library', 'lab', 'settings'];
  const next = views.includes(requested) ? requested : alias[requested] || 'vault';
  const changed = state.view !== next;
  state.view = next;
  views.forEach(id => { $('#view-' + id).hidden = next !== id; });
  $$('.navigation [data-view]').forEach(a => {
    if (a.dataset.view === next) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  if (next === 'settings' && !state.settings) loadSettings().catch(() => {});
  if (changed) {
    const panel = $('#view-' + next);
    panel.classList.add('view-arrival');
    window.scrollTo(0, 0);
    $('h1', panel).focus({ preventScroll: true });
  }
}

function go(view) {
  if (location.hash !== '#/' + view) location.hash = '#/' + view;
  showView();
}

function trace(event, control) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const box = control.getBoundingClientRect(), mark = document.createElement('span');
  mark.className = 'seed-trace';
  mark.setAttribute('aria-hidden', 'true');
  mark.style.left = (event.detail ? event.clientX : box.left + box.width / 2) + 'px';
  mark.style.top = (event.detail ? event.clientY : box.top + box.height / 2) + 'px';
  document.body.append(mark);
  mark.addEventListener('animationend', () => mark.remove(), { once: true });
  setTimeout(() => mark.remove(), 800);
}

// ---------------------------------------------------------------- 首页
function renderHome() {
  if (!state.status) return;
  const counts = state.status.counts;
  say('homeStatus', counts.offline_ready
    ? `${counts.offline_ready} 个模型可试用${counts.active_jobs ? ' · ' + counts.active_jobs + ' 项准备中' : ''} · 权重留在本机`
    : '还没有通过验证的模型，种子库正在按偏好准备。');
}

// ---------------------------------------------------------------- 模型库
const GROUP_ORDER = ['语言', '图像', '动作', '音频', '视频'];
const taskGroups = () => {
  const present = [...new Set(state.catalog.tasks.map(t => t.group))];
  return [...GROUP_ORDER.filter(g => present.includes(g)), ...present.filter(g => !GROUP_ORDER.includes(g))];
};
const taskRank = taskId => {
  const index = state.catalog?.tasks.findIndex(t => t.id === taskId);
  return index < 0 ? 999 : index;
};

function statusOf(c) {
  if (c.seed?.offline_ready && c.seed.integrity === 'ok') return { text: '可试用', cls: 'ready' };
  if (c.seed) return { text: '已落库', cls: 'ready' };
  const job = activeJob(c);
  if (job) {
    return { text: labels[job.state] || job.state_label || job.state,
      cls: working(job) ? 'warn' : job.state === 'failed' ? 'failed' : '' };
  }
  if (c.state === 'gated') return { text: '待授权', cls: '' };
  if (c.install_block) {
    const text = c.install_block.includes('许可') ? '许可待确认'
      : c.install_block.includes('授权') ? '待授权'
      : c.install_block.includes('本机试用') ? '暂不支持本机试用' : c.install_block;
    return { text, cls: '' };
  }
  return { text: '待探索', cls: '' };
}

function repoLink(c) {
  return c.source === 'huggingface' ? new URL('/' + c.repo, HF_BASE).href : '';
}

function sourceLink(url) {
  try {
    const link = new URL(url);
    return link.protocol === 'https:' && ['huggingface.co', 'github.com', 'developers.google.com'].includes(link.hostname) ? link.href : '';
  } catch (_) { return ''; }
}

function modelLinks(c) {
  const links = [];
  const repo = repoLink(c);
  if (repo) links.push(`<a class="text-link" href="${esc(repo)}" target="_blank" rel="noopener noreferrer">来源仓库 ↗</a>`);
  (c.evidence || []).slice(0, 1).forEach(e => {
    const url = sourceLink(e.url);
    if (url) links.push(`<a class="text-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(e.title || '公开证据')} ↗</a>`);
  });
  return links.join('');
}

function modelActions(c, bucket) {
  if (bucket === 'stored') {
    return ready(c)
      ? `<button class="button primary" type="button" data-try="${esc(c.seed.id)}">试用</button>`
      : '<span class="muted small-copy">已保存，等待本机验证</span>';
  }
  if (bucket === 'pending') {
    const job = c.jobs?.[0];
    if (job?.state === 'failed') return `<button class="button outline" type="button" data-download="${esc(c.id)}" data-dmode="local">重试</button>`;
    return '<span class="muted small-copy">准备中</span>';
  }
  const actions = [];
  if (!c.install_block) {
    actions.push(`<button class="button primary" type="button" data-download="${esc(c.id)}" data-dmode="local">下载</button>`);
  }
  if (c.collectable && c.state !== 'gated') {
    actions.push(`<button class="button outline" type="button" data-download="${esc(c.id)}" data-dmode="sota">收藏 SOTA</button>`);
  }
  return actions.join('') || '<span class="muted small-copy">暂不可下载</span>';
}

function modelCard(c, bucket) {
  const task = taskInfo(c.task);
  const status = statusOf(c);
  const job = activeJob(c) || c.jobs?.[0];
  const metas = [];
  if (c.size_bytes) metas.push(size(c.size_bytes));
  if (c.license) metas.push(c.license);
  if (c.estimate?.est_ram_mib > 0) metas.push(`预估内存 ${mib(c.estimate.est_ram_mib)}`);
  if (c.estimate?.est_vram_mib > 0) metas.push(`预估显存 ${mib(c.estimate.est_vram_mib)}`);
  const m = c.measured?.[0];
  if (m) metas.push(`实测 ${m.quality}${m.median_latency_ms ? ` · ${(m.median_latency_ms / 1000).toFixed(1)}s` : ''}`);
  const note = c.rationale || c.reason || (c.seed ? '文件已按固定版本保存到本机。' : '来自公开元数据的候选；尚未下载权重。');
  const reason = c.rationale && c.reason ? `<p class="model-reason">${esc(c.reason)}</p>` : '';
  const defProfile = profilesFor(c.task)[0];
  const pref = preferenceFor(c.task);
  const prefList = profilesFor(c.task);
  const prefLabel = prefList.length > 1
    ? `${profileLabels[prefList[0]] || prefList[0]} +${prefList.length - 1}`
    : (profileLabels[defProfile] || defProfile);
  const chips = [];
  if (defProfile) chips.push(`<span class="chip plain">偏好 · ${esc(prefLabel)}</span>`);
  if (pref.sota_enabled) chips.push('<span class="chip plain">SOTA</span>');
  if (c.sota?.[0]) chips.push(`<span class="chip">领先 · ${esc(c.sota[0].metric)} ${esc(c.sota[0].value)}</span>`);
  const path = c.seed?.rel_dir ? `<code class="seed-path">${esc(state.settings?.seeds_dir || 'seeds')}\\${esc(c.seed.rel_dir)}</code>` : '';
  return `<article class="model-card${ready(c) ? ' is-ready' : ''}">
    <header class="model-card-head">
      <span class="task-chip">${esc(task?.label || c.task)}</span>
      <span class="status ${status.cls}">${esc(status.text)}</span>
    </header>
    <div class="model-title"><h3>${esc(c.variant.replace(/\.gguf$|\.task$/i, ''))}</h3>${chips.join('')}</div>
    <p class="model-repo">${esc(c.repo)}${metas.length ? ' · ' + metas.map(esc).join(' · ') : ''}</p>
    <p class="model-note">${esc(note)}</p>
    ${reason}
    ${job && working(job) ? `<progress max="100" value="${pct(job.progress)}" aria-label="准备进度"></progress>` : ''}
    ${path}
    <footer class="model-card-foot"><div class="model-links">${modelLinks(c)}</div><div class="actions">${modelActions(c, bucket)}</div></footer>
  </article>`;
}

function renderLibrary() {
  if (!state.catalog) return;
  const buckets = [
    { id: 'explore', label: '待探索', items: [] },
    { id: 'pending', label: '待下载', items: [] },
    { id: 'stored', label: '已落库', items: [] },
  ];
  const byId = Object.fromEntries(buckets.map(b => [b.id, b]));
  // 同一仓库 + 变体可能同时存在受控占位与固定修订记录；只保留信息更完整的一条。
  const unique = new Map();
  const score = c => Number(Boolean(c.seed)) * 8 + Number(Boolean(c.jobs?.length)) * 4
    + Number(c.commit_rev && c.commit_rev !== 'main') * 2 + Number(c.size_bytes > 0);
  (state.catalog.candidates || []).forEach(c => {
    const key = `${c.repo}|${c.variant}|${c.task}`;
    const prev = unique.get(key);
    if (!prev || score(c) > score(prev)) unique.set(key, c);
  });
  unique.forEach(c => {
    if (c.seed) byId.stored.items.push(c);
    else if (activeJob(c)) byId.pending.items.push(c);
    else byId.explore.items.push(c);
  });
  buckets.forEach(b => b.items.sort((a, z) => taskRank(a.task) - taskRank(z.task)
    || String(a.variant).localeCompare(String(z.variant), 'zh-CN')));
  const counts = state.status?.counts;
  say('libraryStatus', counts
    ? `${counts.offline_ready} 个可试用 · ${counts.active_jobs} 项准备中 · 共 ${buckets.reduce((n, b) => n + b.items.length, 0)} 条记录`
    : '正在读取种子库');
  render('librarySections', buckets.map(b => `<section class="library-section" aria-labelledby="bucket-${b.id}">
    <header class="bucket-head"><h2 id="bucket-${b.id}">${b.label}</h2><span>${b.items.length}</span></header>
    ${b.items.length ? `<div class="model-grid">${b.items.map(c => modelCard(c, b.id)).join('')}</div>`
      : `<p class="bucket-empty">${b.id === 'stored' ? '还没有落库的模型。' : b.id === 'pending' ? '没有正在准备的任务。' : '没有等待处理的候选。'}</p>`}
  </section>`).join(''));
}

async function download(id, mode) {
  if (state.pending.has(id)) return;
  const candidate = state.catalog.candidates.find(c => c.id === id);
  if (!candidate) return;
  state.pending.add(id);
  renderLibrary();
  try {
    await api('/api/downloads', 'POST', { candidate_id: id, task: candidate.task, mode,
      profile: mode === 'sota' ? 'balanced' : profilesFor(candidate.task)[0] || 'balanced' });
    toast(mode === 'sota' ? '已加入收藏队列' : '已加入下载队列，种子库会自动准备');
    await refresh();
  } catch (e) { toast(e.message); }
  finally { state.pending.delete(id); renderLibrary(); }
}

// ---------------------------------------------------------------- 配置
function settingsPayload(overrides = {}) {
  const s = state.settings || {};
  return {
    auto_update: Boolean(s.auto_update),
    schedule_enabled: Boolean(s.schedule_enabled),
    schedule_time: s.schedule_time || '03:00',
    agent_enabled: s.agent_enabled !== false,
    trusted_publishers: s.trusted_publishers || [],
    manual_repos: s.manual_repos || [],
    subscriptions: s.subscriptions || {},
    simulated_offline: Boolean(s.simulated_offline),
    agent_base_url: s.agent_base_url || '',
    agent_model: s.agent_model || '',
    download_root: s.download_root || '',
    ...overrides,
  };
}

async function loadSettings() {
  const [settings, preferences, agentConfig, schedule] = await Promise.all([
    api('/api/settings'), api('/api/preferences'), api('/api/agent/config'), api('/api/schedule'),
  ]);
  state.settings = settings;
  state.preferences = preferences;
  state.agentConfig = agentConfig;
  state.schedule = schedule;
  renderSettings();
  renderLibrary();
}

function renderSettings() {
  if (!state.settings) return;
  const s = state.settings, a = state.agentConfig || {};
  say('serviceMode', a.configured ? '模型服务已连接' : '使用内置规则');
  say('localEndpoint', `本机模型接口：${location.origin}/v1 · 本机 API Key 保存在 data/api-key`);
  $('#serviceMode').className = 'status ' + (a.configured ? 'ready' : '');
  $('#agentKey').placeholder = a.configured ? '已保存；留空保持不变' : '输入密钥（可选）';
  $('#removeAgentKey').hidden = !a.configured;
  if (document.activeElement !== $('#agentBaseUrl')) $('#agentBaseUrl').value = s.agent_base_url || '';
  if (document.activeElement !== $('#agentModel')) $('#agentModel').value = s.agent_model || '';
  say('updateBadge', s.auto_update && s.schedule_enabled ? '自动更新已开启' : '仅手动更新');
  $('#updateBadge').className = 'status ' + (s.auto_update && s.schedule_enabled ? 'ready' : '');
  $('#autoUpdate').checked = Boolean(s.auto_update && s.schedule_enabled);
  $('#scheduleTime').value = s.schedule_time || '03:00';
  const schedule = state.schedule || {};
  const last = s.effective?.last_attempt || schedule.last_attempt || state.status?.last_sync?.attempt;
  const next = s.effective?.next_run || schedule.next_run;
  say('syncStatus', last ? `上次检查：${new Date(last).toLocaleString('zh-CN', { hour12: false })}` : '还没有检查记录。');
  say('scheduleStatus', s.auto_update && s.schedule_enabled
    ? (next ? `下次计划：${new Date(next).toLocaleString('zh-CN', { hour12: false })}` : '计划已开启，等待本机计划任务注册。')
    : '自动更新已关闭，可随时点“立即更新”。');
  say('storageBadge', s.download_root ? '自定义目录' : '默认目录');
  $('#storageBadge').className = 'status ' + (s.download_root ? 'warn' : 'ready');
  if (document.activeElement !== $('#downloadRoot')) $('#downloadRoot').value = s.download_root || '';
  $('#downloadRoot').placeholder = `留空使用默认目录：${s.default_seeds_dir}`;
  say('seedsDir', `当前模型目录：${s.seeds_dir}`);
  renderPreferences();
}

function preferenceRow(task) {
  const pref = preferenceFor(task.id);
  const profiles = profilesFor(task.id);
  const auto = pref.mode === 'local';
  const sota = Boolean(pref.sota_enabled);
  return `<div class="pref-row" data-task="${esc(task.id)}">
    <div class="pref-name"><strong>${esc(task.label)}</strong><small>${auto ? '自动下载' : sota ? '仅 SOTA' : '未启用'}</small></div>
    <label class="pref-auto"><input type="checkbox" data-pref="auto" ${auto ? 'checked' : ''}><span>自动</span></label>
    <div class="pref-profiles">${['speed', 'balanced', 'accuracy'].map(p =>
      `<label><input type="checkbox" data-pref="profile" value="${p}" ${profiles.includes(p) ? 'checked' : ''}><span>${esc(profileLabels[p])}</span></label>`).join('')}</div>
    <label class="pref-sota"><input type="checkbox" data-pref="sota" ${sota ? 'checked' : ''}><span>SOTA</span></label>
  </div>`;
}

function renderPreferences() {
  if (!state.catalog) return;
  const groups = taskGroups();
  render('preferenceRows', groups.map(group => `<div class="pref-group">
    <p class="pref-group-label">${esc(group)}</p>
    ${state.catalog.tasks.filter(t => t.group === group).map(preferenceRow).join('')}
  </div>`).join(''));
  const count = state.catalog.tasks.length;
  say('preferenceCount', `${count} 个类型 · 默认平衡`);
  $('#preferenceCount').className = 'status';
  say('preferenceStatus', state.prefDirty ? '有未保存的修改' : '');
}

async function savePreferences() {
  const button = $('#savePreferences');
  button.disabled = true;
  say('preferenceStatus', '正在保存…');
  try {
    const rows = $$('#preferenceRows .pref-row');
    for (const row of rows) {
      const task = row.dataset.task;
      const auto = $('[data-pref="auto"]', row).checked;
      const profiles = $$('[data-pref="profile"]:checked', row).map(box => box.value);
      const sota = $('[data-pref="sota"]', row).checked;
      const selected = profiles.length ? profiles : ['balanced'];
      state.preferences[task] = await api('/api/preferences/' + encodeURIComponent(task), 'PUT', {
        mode: auto ? 'local' : 'off', profiles: selected, sota_enabled: sota,
      });
    }
    state.prefDirty = false;
    renderSettings();
    renderLibrary();
    say('preferenceStatus', `已保存 ${rows.length} 个类型`);
    toast('下载偏好已保存');
  } catch (e) {
    say('preferenceStatus', e.message);
  } finally { button.disabled = false; }
}

async function saveService(event) {
  event.preventDefault();
  const button = $('button[type="submit"]', event.target);
  const key = $('#agentKey').value.trim();
  button.disabled = true;
  say('serviceStatus', '正在保存…');
  try {
    state.settings = await api('/api/settings', 'PUT', settingsPayload({
      agent_base_url: $('#agentBaseUrl').value.trim(),
      agent_model: $('#agentModel').value.trim(),
    }));
    if (key) {
      state.agentConfig = await api('/api/agent/config', 'PUT', { api_key: key });
      $('#agentKey').value = '';
      say('serviceStatus', '密钥已保存，正在测试连接…');
      await api('/api/agent/test', 'POST', {});
      say('serviceStatus', '连接成功');
    } else {
      say('serviceStatus', '设置已保存');
    }
    state.schedule = await api('/api/schedule');
    state.agentConfig = await api('/api/agent/config');
    renderSettings();
  } catch (e) { say('serviceStatus', e.message); }
  finally { button.disabled = false; }
}

async function saveUpdate(event) {
  event.preventDefault();
  const button = $('button[type="submit"]', event.target);
  const enabled = $('#autoUpdate').checked;
  button.disabled = true;
  say('syncStatus', '正在保存…');
  try {
    state.settings = await api('/api/settings', 'PUT', settingsPayload({
      auto_update: enabled, schedule_enabled: enabled,
      schedule_time: $('#scheduleTime').value || '03:00',
    }));
    state.schedule = await api('/api/schedule');
    renderSettings();
    toast(enabled ? '已开启每日自动更新' : '已关闭自动更新');
  } catch (e) { say('syncStatus', e.message); }
  finally { button.disabled = false; }
}

async function manualSync() {
  if (state.sync) return;
  state.sync = true;
  $('#manualSync').disabled = true;
  say('manualSync', '正在更新…');
  say('syncStatus', '正在检查公开来源…');
  try {
    const report = await api('/api/sync', 'POST');
    toast(report.status === 'failed' ? '本轮更新未完成'
      : report.status === 'skipped' ? '本轮更新已跳过'
      : report.status === 'partial' ? '已更新可达来源，保留此前目录'
      : report.jobs.length ? '已更新，模型正在准备' : '目录已更新');
    await refresh();
    state.settings = await api('/api/settings');
    state.schedule = await api('/api/schedule');
    renderSettings();
  } catch (e) { say('syncStatus', e.message); }
  finally { state.sync = false; $('#manualSync').disabled = false; say('manualSync', '立即更新'); }
}

async function saveStorage() {
  const raw = $('#downloadRoot').value.trim();
  say('pathStatus', '正在保存…');
  try {
    state.settings = await api('/api/settings', 'PUT', settingsPayload({ download_root: raw }));
    state.schedule = await api('/api/schedule');
    renderSettings();
    say('pathStatus', '已保存');
    toast(raw ? '模型保存位置已更新' : '已恢复默认保存位置');
    await refresh();
  } catch (e) { say('pathStatus', e.message); }
}

// ---------------------------------------------------------------- 试用
function trialCopy(task) {
  return taskField(task, 'trial') || taskField(task, 'description');
}

function renderLab() {
  if (!state.catalog) return;
  const available = state.status?.available_now || [];
  if (!state.labTask || !taskInfo(state.labTask)) {
    state.labTask = available.find(m => m.task === 'chat')?.task || available[0]?.task || 'chat';
  }
  const task = taskInfo(state.labTask);
  const items = available.filter(m => m.task === task.id);
  if (!items.some(m => m.seed_id === state.model)) {
    const preferred = state.catalog.defaults.find(p => p.task === task.id && p.profile === 'balanced');
    state.model = items.find(m => m.seed_id === preferred?.seed_id)?.seed_id || items[0]?.seed_id || '';
  }
  const usable = state.catalog.tasks.filter(t => t.enabled || t.id === task.id);
  const groups = taskGroups().filter(g => usable.some(t => t.group === g));
  const pendingCount = state.catalog.tasks.length - usable.length;
  render('labTasks', groups.map(group => `<div class="lab-group"><p>${esc(group)}</p><div class="lab-chips">${usable.filter(t => t.group === group).map(t => {
    const n = available.filter(m => m.task === t.id).length;
    const meta = n ? `<small aria-label="${n} 个可试用">${n}</small>` : '<small class="none">无可用</small>';
    return `<button type="button" class="lab-chip" data-lab-task="${esc(t.id)}" aria-pressed="${t.id === task.id}">${esc(t.label)}${meta}</button>`;
  }).join('')}</div></div>`).join('') +
    (pendingCount ? `<p class="lab-note">另有 ${pendingCount} 个类型仍在接入，可在<a href="#/library" data-view="library">模型库</a>查看来源与收藏。</p>` : ''));
  say('labTaskTitle', task.label);
  say('labTaskHint', task.enabled ? trialCopy(task) : '此类型的本机试用仍在接入');
  $('#labModelWrap').hidden = !task.enabled;
  render('labModel', items.map(m => `<option value="${esc(m.seed_id)}">${esc(m.variant.replace(/\.gguf$|\.task$/i, ''))}</option>`).join('') || '<option value="">暂无离线版本</option>');
  $('#labModel').value = state.model;
  const busy = Boolean(state.chat || state.mediaRunning);
  $('#labModel').disabled = busy || !state.model;
  const chat = task.id === 'chat', media = task.id === 'pose' || task.id === 'hand';
  const has = task.enabled && Boolean(state.model);
  $('#labEmpty').hidden = has;
  $('#chatPanel').hidden = !has || !chat;
  $('#mediaForm').hidden = !has || !media;
  $('#newChat').hidden = !has || !chat;
  $('#newChat').disabled = busy;
  $('#sendChat').disabled = busy;
  $('#stopChat').hidden = !state.chat;
  $('#runMedia').disabled = busy || !state.image;
  $('#mediaFile').disabled = busy;
  if (media) say('runMedia', task.id === 'hand' ? '识别手部关键点' : '识别人体关键点');
  if (!task.enabled) {
    say('labEmptyTitle', '这个类型的试用待接入');
    say('labEmptyText', (taskField(task, 'description') || '此类型还没有登记本机运行器。') + '可以先在模型库查看来源或收藏权重。');
  } else if (!state.model) {
    const preparing = state.catalog.candidates.some(c => c.task === task.id && c.jobs?.some(working));
    say('labEmptyTitle', preparing ? '正在准备本机版本' : '还没有可试用的版本');
    say('labEmptyText', '种子准备好后会自动出现在这里。');
  }
}

function clearTrial() {
  state.messages = [];
  $('#chatLog').replaceChildren();
  $('#chatInput').value = '';
  $('#mediaFile').value = '';
  $('#mediaName').textContent = '未选择文件';
  state.fileVersion++;
  state.image?.bitmap.close();
  state.image = null;
  $('#mediaPreview').hidden = true;
  say('runInfo', '本地会话');
}

function openTrial(id) {
  const item = state.status?.available_now.find(m => m.seed_id === id);
  if (!item) return;
  if (state.chat || state.mediaRunning) { toast('请等当前试用完成'); return; }
  if (state.model !== id) clearTrial();
  state.labTask = item.task;
  state.model = id;
  renderLab();
  go('lab');
}

function message(who, body, role) {
  const row = document.createElement('div'), tag = document.createElement('span'), value = document.createElement('p');
  row.className = 'msg ' + role;
  tag.className = 'who';
  tag.textContent = who;
  value.className = 'message';
  value.textContent = body;
  row.append(tag, value);
  $('#chatLog').append(row);
  row.scrollIntoView({ block: 'nearest' });
  return value;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $('#chatInput'), prompt = input.value.trim();
  if (!prompt || !state.model || state.chat) return;
  const controller = new AbortController();
  state.chat = controller;
  input.value = '';
  const user = { role: 'user', content: prompt };
  message('你', prompt, 'user');
  const output = message('本地模型', '正在准备回复…', 'assistant');
  renderLab();
  say('runInfo', '本机生成中');
  let reply = '', done = false;
  try {
    const response = await api('/api/infer', 'POST', { task: 'chat', model: state.model,
      inputs: { messages: [...state.messages, user] }, stream: true }, controller.signal, true);
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let pending = '';
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') { done = true; continue; }
        if (!payload) continue;
        const item = JSON.parse(payload);
        if (item.error) throw new Error(item.error.message || '本地推理未完成');
        reply += item.choices?.[0]?.delta?.content || '';
        if (reply) {
          output.textContent = reply;
          $('#chatLog').scrollTop = $('#chatLog').scrollHeight;
        }
      }
    }
    if (!done || !reply) throw new Error('没有收到完整回复');
    state.messages.push(user, { role: 'assistant', content: reply });
    say('runInfo', '已完成');
  } catch (e) {
    const stopped = controller.signal.aborted;
    output.textContent = [reply, stopped ? '已停止接收' : e.message].filter(Boolean).join('\n\n');
    input.value = prompt;
    say('runInfo', stopped ? '已停止接收' : '本地试用未完成');
  } finally { state.chat = null; renderLab(); }
}

function draw(points = []) {
  const canvas = $('#mediaPreview'), image = state.image.bitmap;
  const scale = Math.min(1, 1000 / Math.max(image.width, image.height));
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#b9df9d';
  ctx.strokeStyle = '#101715';
  ctx.lineWidth = 1.5;
  for (const group of points) for (const p of group) {
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();
  }
  canvas.hidden = false;
}

async function selectImage() {
  const file = $('#mediaFile').files[0], version = ++state.fileVersion;
  state.image?.bitmap.close();
  state.image = null;
  $('#mediaPreview').hidden = true;
  say('mediaName', file ? file.name : '未选择文件');
  renderLab();
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    $('#mediaFile').value = '';
    say('mediaName', '未选择文件');
    say('runInfo', '图片超过 8 MiB');
    return;
  }
  try {
    const bitmap = await createImageBitmap(file);
    if (version !== state.fileVersion) { bitmap.close(); return; }
    if (bitmap.width * bitmap.height > 16_000_000) { bitmap.close(); throw new Error('图片尺寸过大'); }
    state.image = { bitmap, file };
    draw();
    say('runInfo', '已选择 ' + file.name);
  } catch (e) {
    if (version === state.fileVersion) {
      say('runInfo', '无法读取图片');
      say('mediaName', '未选择文件');
    }
  }
  renderLab();
}

async function runMedia(event) {
  event.preventDefault();
  if (!state.image || state.mediaRunning) return;
  const task = state.labTask, model = state.model;
  state.mediaRunning = true;
  renderLab();
  say('runInfo', '本机识别中');
  try {
    const url = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('无法读取图片'));
      reader.readAsDataURL(state.image.file);
    });
    const result = await api('/api/infer', 'POST', { task, model,
      inputs: { image: String(url).split(',')[1] } });
    const points = result.outputs.landmarks;
    draw(points);
    say('runInfo', points.length ? `识别到 ${points.length} 组 · ${points.flat().length} 个关键点` : '未识别到关键点');
  } catch (e) { say('runInfo', e.message); }
  finally { state.mediaRunning = false; renderLab(); }
}

// ---------------------------------------------------------------- 数据刷新
function refresh() {
  if (state.refresh) return state.refresh;
  const firstCatalog = !state.catalog;
  state.refresh = Promise.all([api('/api/catalog'), api('/api/status'), api('/api/jobs')]).then(([catalog, status, jobs]) => {
    state.catalog = catalog;
    state.status = status;
    say('connectionText', status.network.simulated_offline ? '更新已暂停' : status.network.source_error ? '部分来源不可达' : '本机已连接');
    $('.connection').dataset.state = status.network.source_error ? 'error' : status.network.simulated_offline ? 'paused' : 'ok';
    renderHome();
    renderLibrary();
    renderLab();
    if (firstCatalog && state.settings) renderSettings();
  }).finally(() => { state.refresh = null; });
  return state.refresh;
}

// ---------------------------------------------------------------- 事件
document.addEventListener('click', event => {
  const el = event.target.closest('button,a');
  if (!el || el.disabled || (el.tagName === 'A' && (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey))) return;
  if (el.matches('.navigation a,.button.primary,.model-card')) trace(event, el);
  if (el.dataset.view) {
    event.preventDefault();
    go(el.dataset.view);
  } else if (el.dataset.labTask) {
    if (el.dataset.labTask !== state.labTask) {
      state.labTask = el.dataset.labTask;
      state.model = '';
      clearTrial();
      renderLab();
    }
  } else if (el.dataset.try) openTrial(el.dataset.try);
  else if (el.dataset.download) download(el.dataset.download, el.dataset.dmode || 'local');
});
window.addEventListener('hashchange', showView);
document.addEventListener('animationend', event => {
  if (event.animationName === 'view-arrival') event.target.classList.remove('view-arrival');
});

$('#preferenceRows').addEventListener('change', event => {
  const row = event.target.closest('.pref-row');
  if (!row) return;
  if (event.target.dataset.pref === 'auto' && event.target.checked
      && !$$('[data-pref="profile"]:checked', row).length) {
    $('[data-pref="profile"][value="balanced"]', row).checked = true;
  }
  const auto = $('[data-pref="auto"]', row).checked;
  const sota = $('[data-pref="sota"]', row).checked;
  $('small', row).textContent = auto ? '自动下载' : sota ? '仅 SOTA' : '未启用';
  state.prefDirty = true;
  say('preferenceStatus', '有未保存的修改');
});

$('#serviceForm').addEventListener('submit', saveService);
$('#updateForm').addEventListener('submit', saveUpdate);
$('#storageForm').addEventListener('submit', event => { event.preventDefault(); saveStorage(); });
$('#manualSync').addEventListener('click', manualSync);
$('#resetPath').addEventListener('click', () => { $('#downloadRoot').value = ''; saveStorage(); });
$('#savePreferences').addEventListener('click', savePreferences);
$('#removeAgentKey').addEventListener('click', async () => {
  try {
    state.agentConfig = await api('/api/agent/config', 'DELETE');
    state.settings = await api('/api/settings');
    renderSettings();
    say('serviceStatus', state.agentConfig.configured ? '已移除本机保存的密钥，环境变量仍提供密钥' : '已移除本机保存的密钥');
  } catch (e) { say('serviceStatus', e.message); }
});

$('#labModel').addEventListener('change', e => { state.model = e.target.value; clearTrial(); renderLab(); });
$('#newChat').addEventListener('click', () => { clearTrial(); $('#chatInput').focus(); });
$('#chatForm').addEventListener('submit', sendChat);
$('#stopChat').addEventListener('click', () => state.chat?.abort());
$('#mediaFile').addEventListener('change', selectImage);
$('#mediaForm').addEventListener('submit', runMedia);

async function connect() {
  try {
    if (!state.session) state.session = (await api('/api/session')).session_token;
    await refresh();
    if (!state.preferences) {
      state.preferences = await api('/api/preferences');
      renderLibrary();
      if (state.settings) renderSettings();
    }
    if (!state.settings && (state.view === 'settings' || !state.initialized)) await loadSettings();
    state.initialized = true;
  } catch (e) {
    say('connectionText', '本机服务暂未连接');
    if (!state.catalog) say('libraryStatus', '正在等待本机服务');
  }
}
showView();
connect();
setInterval(connect, 4500);
