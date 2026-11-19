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
  labFilter: '', labTask: null, model: '', view: 'vault', refresh: null,
  sync: false, pending: new Set(), toast: 0, chat: null, messages: [],
  image: null, fileVersion: 0, mediaRunning: false, initialized: false,
  prefDirty: false, libraryStage: 'explore', selected: new Map(), submitting: false,
};
const labels = {
  queued: '排队中', downloading: '下载中', verifying: '校验中',
  debugging: '准备环境', retesting: '本机验证中', waiting_network: '等待来源',
  waiting_resource: '等待本机资源', failed: '准备未完成',
};
const profileLabels = { speed: '速度优先', balanced: '平衡', accuracy: '精度优先', sota: 'SOTA' };
const taskCopy = {
  pose: { description: '识别本地图片中的人体关键点，输出坐标与预览叠层', trial: '选择一张本地图片，输出人体关键点与预览叠层' },
  hand: { description: '识别本地图片中的手部关节，输出坐标与预览叠层', trial: '选择一张本地图片，输出每只手的关节坐标与预览叠层' },
};
const taskInfo = id => state.catalog?.tasks.find(t => t.id === id);
const taskField = (task, field) => task?.[field] || taskCopy[task?.id]?.[field] || '';
const ready = c => Boolean(c.seed?.offline_ready && c.seed.integrity === 'ok');
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

let passage = null;

async function showView(animate = true) {
  if (location.hash && !location.hash.startsWith('#/')) return;
  const views = ['vault', 'library', 'lab', 'settings'];
  const alias = { discover: 'library', agent: 'settings' };
  const requested = location.hash.slice(2);
  const next = views.includes(requested) ? requested : Object.hasOwn(alias, requested) ? alias[requested] : 'vault';
  if (passage?.to === next) return;
  passage?.stop();

  const from = state.view;
  if (from === next) return;
  const panel = $('#view-' + next);
  const scene = $('.cave-scene');
  const blackout = $('.cave-blackout');
  const main = $('#main');
  const nav = $('.masthead');
  const motions = [];
  const trip = { to: next, stop() {
    motions.forEach(motion => motion.cancel());
    main.inert = nav.inert = false;
    delete scene.dataset.mouth;
    passage = null;
  } };
  passage = trip;

  function play(element, frames, timing) {
    const motion = element.animate(frames, { fill: 'both', ...timing });
    motions.push(motion);
    return motion;
  }
  function travel(mouth, direction) {
    scene.dataset.mouth = mouth;
    const rock = $('.gate[data-view="' + mouth + '"] .boulder', scene);
    const timing = { duration: 1050, direction };
    play(rock, [
      { transform: 'none', easing: 'ease-in-out' },
      { transform: 'translateX(var(--roll)) rotate(var(--tilt))', offset: .62 },
      { transform: 'translateX(var(--roll)) rotate(var(--tilt))' },
    ], timing);
    return play(scene, [
      { transform: 'none' },
      { transform: 'none', offset: .32, easing: 'cubic-bezier(.5,.04,.72,1)' },
      { transform: 'translate(calc(50% - var(--mouth-x)),calc(50% - var(--mouth-y))) scale(6)' },
    ], timing);
  }
  const moving = animate && !matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fade = (start, end, duration, delay = 0) => play(blackout, [{ opacity: start }, { opacity: end }], { duration, delay });
  try {
    if (moving) {
      main.inert = nav.inert = true;
      if (from === 'vault') travel(next, 'normal');
      await fade(0, 1, from === 'vault' ? 290 : 320, from === 'vault' ? 760 : 0).finished;
    }
    state.view = next;
    views.forEach(id => { $('#view-' + id).hidden = next !== id; });
    if (next === 'settings' && !state.settings) loadSettings().catch(() => {});
    if (moving) {
      const retreat = next === 'vault' && travel(from, 'reverse');
      await fade(1, 0, retreat ? 290 : 380).finished;
      if (retreat) await retreat.finished;
    }
  } catch (error) {
    if (error.name !== 'AbortError') throw error;
  } finally {
    if (passage === trip) {
      trip.stop();
      const focusTarget = next === 'vault' ? $('.gate[data-view="' + from + '"]') : $('h1', panel);
      focusTarget.focus({ preventScroll: true });
    }
  }
}

function go(view) {
  location.hash = '#/' + view;
}

// ---------------------------------------------------------------- 首页
function renderHome() {
  const counts = state.status?.counts;
  say('homeStored', counts?.seeds ?? '—');
  say('homeReady', counts?.offline_ready ?? '—');
  say('homeJobs', counts?.active_jobs ?? '—');
}

// ---------------------------------------------------------------- 模型库
const GROUP_ORDER = ['语言', '图像', '动作', '音频', '视频'];
const taskGroups = () => {
  const present = [...new Set(state.catalog.tasks.map(t => t.group))];
  return [...GROUP_ORDER.filter(g => present.includes(g)), ...present.filter(g => !GROUP_ORDER.includes(g))];
};
function taskOptions() {
  return '<option value="">全部类型</option>' + taskGroups().map(group =>
    `<optgroup label="${esc(group || '其他')}">${state.catalog.tasks.filter(t => t.group === group)
      .map(t => `<option value="${esc(t.id)}">${esc(t.label)}</option>`).join('')}</optgroup>`).join('');
}
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
  return c.source === 'huggingface' && /^[\w.-]+\/[\w.-]+$/.test(c.repo)
    ? new URL('/' + c.repo + '/tree/' + encodeURIComponent(c.commit_rev), HF_BASE).href : '';
}

function sourceLink(url) {
  try {
    const link = new URL(url);
    return link.protocol === 'https:' && ['huggingface.co', 'github.com', 'developers.google.com', 'ai.google.dev', 'arxiv.org', 'arena.ai', 'artificialanalysis.ai', 'www2.statmt.org', 'aclanthology.org', 'openaccess.thecvf.com'].includes(link.hostname) ? link.href : '';
  } catch (_) { return ''; }
}

function modelLinks(c) {
  const links = [];
  const repo = sourceLink(c.release_url) || repoLink(c);
  if (repo) links.push(`<a class="text-link" href="${esc(repo)}" target="_blank" rel="noopener noreferrer">来源仓库 ↗</a>`);
  (c.evidence || []).slice(0, 1).forEach(e => {
    const url = sourceLink(e.url);
    const title = e.title || '公开证据';
    const label = title === '上游模型卡与文件清单' ? '上游文件' : title;
    if (url) links.push(`<a class="text-link" href="${esc(url)}" title="${esc(title)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>`);
  });
  return links.join('');
}

function modelActions(c, bucket) {
  if (c.reference_only) return '<span class="muted small-copy">评测参考</span>';
  if (bucket === 'stored') {
    if (ready(c)) return `<button class="button primary" type="button" data-try="${esc(c.seed.id)}">试用</button>`;
    const job = activeJob(c);
    if (job && working(job) && !job.state.startsWith('waiting_')) return '<span class="muted small-copy">验证中</span>';
    if (!c.install_block) return `<button class="button primary" type="button" data-download="${esc(c.id)}" data-dmode="local">验证本机试用</button>`;
    return '<span class="muted small-copy">已收藏，暂不支持本机试用</span>';
  }
  if (bucket === 'pending') {
    const job = c.jobs?.[0];
    if (job?.state === 'failed' || job?.state.startsWith('waiting_')) return `<button class="button outline" type="button" data-download="${esc(c.id)}" data-dmode="${esc(job.mode || 'local')}">重试</button>`;
    return '<span class="muted small-copy">准备中</span>';
  }
  const mode = selectionMode(c);
  if (!mode) return '<span class="muted small-copy">暂不可下载</span>';
  return `<label class="model-select"><input type="checkbox" data-select-model="${esc(c.id)}" ${state.selected.has(c.id) ? 'checked' : ''} ${state.submitting ? 'disabled' : ''}>勾选下载</label>`;
}

function selectionMode(c) {
  const modes = c.recommended_modes || [];
  const filtered = $('#libraryMode').value;
  return modes.includes(filtered) ? filtered
    : modes.find(mode => mode !== 'sota') || modes[0];
}

function modelScores(c) {
  return [...new Map([...(c.sota || []), ...(c.public_evidence || [])]
    .filter(e => Number.isFinite(e.value) && e.url)
    .map(e => [`${e.benchmark_key || e.title}|${e.metric}`, e])).values()];
}

function modelSummary(c) {
  if (c.brief) return c.brief;
  const task = taskInfo(c.task);
  const score = modelScores(c)[0];
  const measured = c.measured?.[0];
  return {
    ability: taskField(task, 'description') || trialCopy(task) || task?.label || '用途见发布方说明。',
    effect: score ? `${score.title} 的公开成绩为 ${score.value}${score.unit || ''}${score.public_rank ? `，榜单第 ${score.public_rank}` : ''}。`
      : measured ? `历史本机评测通过 ${measured.passed_cases}/${measured.total_cases} 项固定用例。`
      : '目前缺少此版本可核对的公开量化成绩。',
    limit: c.install_block ? '本机试用尚未接入，当前只能归档和查阅资料。'
      : ready(c) ? '已在本机通过固定用例验证。实际表现随输入和运行配置变化。'
      : '此版本尚未通过本机验证，效果和速度取决于输入和运行配置。',
  };
}

function modelGuide(c) {
  const task = taskInfo(c.task);
  const brief = modelSummary(c);
  const resources = [];
  if (c.size_bytes) resources.push(size(c.size_bytes));
  if (c.estimate?.est_ram_mib > 0) resources.push(`预估内存 ${mib(c.estimate.est_ram_mib)}`);
  if (c.estimate?.est_vram_mib > 0) resources.push(`预估显存 ${mib(c.estimate.est_vram_mib)}`);
  const adapter = c.task === 'chat' ? 'llama.cpp' : 'MediaPipe';
  const local = !c.install_block && (c.seed || c.recommended_modes?.some(mode => mode !== 'sota'));
  const guide = c.integration || (local
    ? `计划使用 ${adapter} 在本机运行。${trialCopy(task) || '输入文本并接收回答'}。通过离线验证后，可在试用台使用，也可调用 ${c.task === 'chat' ? '/v1/chat/completions' : '/api/infer'}。`
    : '本机运行方式需另行验证。接入时请以固定版本的上游说明为准。');
  const files = (c.files || []).filter(f => ['model_card', 'license', 'usage'].includes(f.kind));
  const chat = c.task === 'chat';
  const model = c.seed?.id || c.id;
  const payload = chat ? { model, messages: [{ role: 'user', content: '你好' }] }
    : { task: c.task, model, inputs: { image: '<图片的 Base64 内容>' } };
  const inputLabels = { messages: '对话消息', text: '文本', image: '图片', audio: '音频', video: '视频',
    source_image: '来源图片', target_image: '目标图片' };
  const inputs = (task?.inputs || []).map(input => inputLabels[input] || input).join('、');
  const related = [...(c.sota || []), ...(c.public_evidence || [])];
  const results = modelScores(c);
  const evidence = (results.length ? results : related).slice(0, 4).map(e => {
    const url = sourceLink(e.url);
    const title = esc(e.title || e.source || '公开资料');
    const value = Number.isFinite(e.value) ? ` · ${esc(e.metric)} ${esc(e.value)}${esc(e.unit || '')}` : '';
    const rank = e.public_rank ? ` · 榜单第 ${esc(e.public_rank)}` : '';
    return `<li>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title}${value}${rank}${e.note ? `<small>${esc(e.note)}</small>` : ''}</li>`;
  }).join('');
  const measured = c.measured?.[0];
  const upstream = (c.model_card || '').replace(/^---\s*\r?\n[\s\S]*?\r?\n---[ \t]*\r?\n/, '')
    .replace(/<!--[\s\S]*?-->|<\/?[a-z][^>]*>/gi, '')
    .replace(/!\[[^\]]*\]\([^\n)]*\)/g, '')
    .replace(/\[([^\]\n]+)\]\([^\n)]*\)/g, '$1')
    .replace(/(?:\r?\n[ \t]*){3,}/g, '\n\n').trim().slice(0, 5400);
  return `<dl class="model-facts">
      <div><dt>用途</dt><dd>${esc(brief.ability)}</dd></div>
      ${inputs ? `<div><dt>输入</dt><dd>${esc(inputs)}</dd></div>` : ''}
      <div><dt>使用限制</dt><dd>${esc(brief.limit)}</dd></div>
      <div><dt>版本与许可</dt><dd><code>${esc(c.commit_rev)}</code> · ${esc(c.license || '未确认')}</dd></div>
      ${resources.length ? `<div><dt>文件与资源</dt><dd>${resources.map(esc).join(' · ')}</dd></div>` : ''}
    </dl>
    <h4>公开效果</h4>
    ${c.review_method ? `<p>${esc(c.review_method)}</p>` : ''}
    ${evidence ? `<ul>${evidence}</ul>` : ''}
    ${results.length ? '' : `<p>${esc(c.brief?.effect || '暂无绑定到这个变体的公开量化成绩。可查阅上游资料中的示例和使用限制。')}</p>`}
    ${c.alternatives?.length ? `<h4>同组比较</h4><ul>${c.alternatives.map(alternative =>
      `<li><strong>${esc(alternative.variant)}</strong><small>${alternative.evidence.map(e => {
        const url = sourceLink(e.url);
        const label = esc(e.title) + ' ' + esc(e.value) + esc(e.unit || '');
        return url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
      }).join(' · ')}</small></li>`).join('')}</ul>` : ''}
    <h4>本机表现</h4>
    <p>${measured ? `${c.seed ? '实测' : '历史实测'} · 评测版本 ${esc(measured.eval_version)} · 通过 ${esc(measured.passed_cases)}/${esc(measured.total_cases)} · 中位耗时 ${esc(Math.round(measured.median_latency_ms))} ms` : '暂无本机实测。'}</p>
    <h4>运行与接入</h4>
    <p>${esc(guide)}</p>
    ${sourceLink(c.release_url) ? `<p><a href="${esc(sourceLink(c.release_url))}" target="_blank" rel="noopener noreferrer">官方发布与权重</a></p>` : ''}
    ${files.length ? `<p>随包资料 <code>${files.map(f => esc(f.repo_path)).join('、')}</code></p>` : ''}
    ${local ? `<p>验证通过后可发送以下本机请求。Bearer 密钥保存在 <code>data/api-key</code>。</p>
    <pre>${esc('POST ' + (chat ? '/v1/chat/completions' : '/api/infer') + '\n' + JSON.stringify(payload, null, 2))}</pre>` : ''}
    ${upstream ? `<details class="upstream-card"><summary>上游模型卡摘录</summary><pre>${esc(upstream)}</pre></details>` : ''}`;
}

function modelCard(c, bucket) {
  const task = taskInfo(c.task);
  const status = bucket === 'pending' ? statusOf(c) : null;
  const job = activeJob(c) || c.jobs?.[0];
  const name = esc(c.variant.replace(/\.gguf$|\.task$/i, ''));
  const modes = (c.recommended_modes || []).map(mode => esc(profileLabels[mode])).join(' · ');
  return `<article class="model-card">
    <header class="model-card-head">
      <span class="task-chip">${esc(task?.label || c.task)}</span>
      ${status ? `<span class="status ${status.cls}">${esc(status.text)}</span>` : ''}
    </header>
    <h3 title="${name}">${name}</h3>
    ${modes ? `<p class="model-modes">${modes}</p>` : ''}
    ${job && working(job) ? `<progress max="100" value="${pct(job.progress)}" aria-label="准备进度"></progress>` : ''}
    <footer class="model-card-foot plain-actions">
      <button class="model-details text-link" type="button" data-model-details="${esc(c.id)}" aria-haspopup="dialog">模型说明与接入</button>
      ${modelLinks(c)}
      ${modelActions(c, bucket)}
    </footer>
  </article>`;
}

function renderLibrary() {
  if (!state.catalog) return;
  const stage = state.libraryStage;
  const candidates = state.catalog.candidates || [];
  const typeSelect = $('#libraryTask');
  const chosen = typeSelect.value || '';
  render('libraryTask', taskOptions());
  typeSelect.value = chosen;
  const mode = $('#libraryMode').value;
  const selectable = candidates.filter(c => !c.reference_only && !c.seed && !activeJob(c) && c.recommended_modes?.length);
  const ids = new Set(selectable.map(c => c.id));
  for (const id of state.selected.keys()) if (!ids.has(id)) state.selected.delete(id);
  const bucket = c => c.seed ? 'stored' : activeJob(c) ? 'pending' : 'explore';
  const items = candidates.filter(c => bucket(c) === stage && (!chosen || c.task === chosen)
      && (!mode || [...(c.recommended_modes || []), ...(c.download_profiles || [])].includes(mode)))
    .sort((a, b) => taskRank(a.task) - taskRank(b.task)
      || String(a.variant).localeCompare(String(b.variant), 'zh-CN'));
  state.visibleCandidates = stage === 'explore' ? items.filter(c => ids.has(c.id)) : [];
  $('#libraryActions').hidden = stage !== 'explore';
  $('#selectVisible').disabled = state.submitting || !state.visibleCandidates.length;
  say('selectVisible', state.visibleCandidates.length && state.visibleCandidates.every(c => state.selected.has(c.id))
    ? '取消全选' : '全选结果');
  $('#downloadSelected').disabled = state.submitting || !state.selected.size;
  say('downloadSelected', state.submitting ? '正在提交…' : `下载所选（${state.selected.size}）`);
  $$('[data-library-stage]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.libraryStage === stage));
  });
  say('libraryStatus', `${items.length} 个模型`);
  const slots = chosen && stage === 'explore' ? Object.entries(profileLabels)
    .filter(([key]) => !mode || key === mode)
    .map(([key, label]) => {
      const found = candidates.find(c => c.task === chosen && c.recommended_modes?.includes(key));
      const text = found?.reference_only ? '公开评测参考' : found ? { stored: '已落库', pending: '下载中', explore: '待下载' }[bucket(found)] : key === 'sota' ? '暂无可比较的公开成绩' : '暂无合格版本';
      return `<span class="model-slot${found ? ' found' : ''}">${esc(label)} <small>${text}</small></span>`;
    }).join('') : '';
  const empty = { stored: '这里还没有已落库的模型。', pending: '没有匹配的下载作业。', explore: mode === 'sota' ? '暂无匹配的公开领先记录。榜单筛选不限制本机算力。' : '没有匹配的待下载模型。' };
  render('librarySections', (slots ? `<div class="model-slots">${slots}</div>` : '') +
    (items.length ? `<div class="model-grid">${items.map(c => modelCard(c, stage)).join('')}</div>`
      : `<p class="bucket-empty">${empty[stage]}</p>`));
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

async function downloadSelection() {
  if (state.submitting || !state.selected.size) return;
  state.submitting = true;
  let submitted = 0;
  const errors = [];
  renderLibrary();
  try {
    for (const [id, profile] of state.selected) {
      const candidate = state.catalog.candidates.find(c => c.id === id);
      try {
        await api('/api/downloads', 'POST', {
          candidate_id: id, task: candidate.task, profile: profile === 'sota' ? 'balanced' : profile,
          mode: profile === 'sota' ? 'sota' : candidate.install_block ? 'requested' : 'local',
        });
        state.selected.delete(id);
        submitted++;
      } catch (e) { errors.push(e.message); }
    }
    await refresh();
    toast(`已提交 ${submitted} 项${errors.length ? `，${errors.length} 项未提交：${errors[0]}` : ''}`);
  } catch (e) { toast(`已提交 ${submitted} 项，目录刷新失败：${e.message}`); }
  finally { state.submitting = false; renderLibrary(); }
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
  say('updateBadge', s.auto_update ? '已开启自动下载' : '手动选择下载');
  $('#updateBadge').className = 'status ' + (s.auto_update && s.schedule_enabled ? 'ready' : '');
  $('#autoUpdate').checked = Boolean(s.auto_update);
  $('#scheduleDiscovery').checked = Boolean(s.schedule_enabled);
  $('#scheduleTime').value = s.schedule_time || '03:00';
  const schedule = state.schedule || {};
  const last = s.effective?.last_attempt || schedule.last_attempt || state.status?.last_sync?.attempt;
  const next = s.effective?.next_run || schedule.next_run;
  say('syncStatus', last ? `上次检查：${new Date(last).toLocaleString('zh-CN', { hour12: false })}` : '还没有检查记录。');
  say('scheduleStatus', s.schedule_enabled
    ? (next ? `下次计划：${new Date(next).toLocaleString('zh-CN', { hour12: false })}` : '计划已开启，等待本机计划任务注册。')
    : '每日查找已关闭，可随时点“立即查找”。');
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
    state.settings = await api('/api/settings');
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
    }
    if (state.agentConfig?.configured) {
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
      auto_update: enabled, schedule_enabled: $('#scheduleDiscovery').checked,
      schedule_time: $('#scheduleTime').value || '03:00',
    }));
    state.schedule = await api('/api/schedule');
    renderSettings();
    toast(enabled ? '已开启自动下载' : '已关闭自动下载');
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
    const report = await api('/api/sync?catalog_only=true', 'POST');
    toast(report.status === 'failed' ? '本轮更新未完成'
      : report.status === 'skipped' ? '本轮更新已跳过'
      : report.status === 'partial' ? '已更新可达来源，保留此前目录'
      : report.jobs.length ? '已更新，模型正在准备' : '目录已更新');
    await refresh();
    state.settings = await api('/api/settings');
    state.schedule = await api('/api/schedule');
    renderSettings();
  } catch (e) { say('syncStatus', e.message); }
  finally { state.sync = false; $('#manualSync').disabled = false; say('manualSync', '立即查找'); }
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
  if (state.labFilter && !taskInfo(state.labFilter)) state.labFilter = '';
  const items = available.filter(m => !state.labFilter || m.task === state.labFilter);
  const first = items.find(m => m.task === 'chat') || items[0];
  const preferred = state.catalog.defaults.find(p => p.task === first?.task && p.profile === 'balanced');
  const selected = items.find(m => m.seed_id === state.model)
    || items.find(m => m.seed_id === preferred?.seed_id) || first;
  const busy = Boolean(state.chat || state.mediaRunning);
  if (!busy) {
    if (state.model && state.model !== selected?.seed_id) clearTrial();
    state.model = selected?.seed_id || '';
    state.labTask = selected?.task || state.labFilter;
  }
  const task = taskInfo(state.labTask);
  render('labTasks', taskOptions());
  $('#labTasks').value = state.labFilter;
  $('#labTasks').disabled = busy;
  $('#labModelWrap').hidden = false;
  render('labModel', items.map(m => `<option value="${esc(m.seed_id)}">${esc(m.variant.replace(/\.gguf$|\.task$/i, ''))}</option>`).join('') || '<option value="">暂无可试用模型</option>');
  $('#labModel').value = state.model;
  $('#labModel').disabled = busy || !state.model;
  const chat = task?.id === 'chat', media = task?.id === 'pose' || task?.id === 'hand';
  const has = task?.enabled && (chat || media) && Boolean(state.model);
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
  if (task && (!task.enabled || !(chat || media))) {
    say('labEmptyText', `${task.label}尚未接入本机试用，可到模型库查看该类型的模型和资料。`);
  } else if (!has) {
    const preparing = state.catalog.candidates.some(c => (!state.labFilter || c.task === state.labFilter) && c.jobs?.some(working));
    say('labEmptyText', preparing ? '正在准备本机版本。' : '还没有通过本机验证的模型。');
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
  state.labFilter = item.task;
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
  state.refresh = api('/api/status').then(async status => {
    if (firstCatalog || state.catalogRevision !== status.catalog_revision) {
      state.catalog = await api('/api/catalog?curated=true');
      state.catalogRevision = status.catalog_revision;
    }
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
  let el = event.target.closest('button,a');
  if (!el && !event.target.closest('label,input') && !window.getSelection()?.toString()) {
    el = event.target.closest('.model-card')?.querySelector('[data-model-details]');
  }
  if (!el || el.disabled || (el.tagName === 'A' && (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey))) return;
  if (el.dataset.modelDetails) {
    const candidate = state.catalog.candidates.find(c => c.id === el.dataset.modelDetails);
    if (candidate) {
      const dialog = $('#modelDialog');
      dialog.dataset.modelId = candidate.id;
      say('modelDialogTitle', candidate.variant);
      render('modelDialogBody', modelGuide(candidate));
      dialog.showModal();
      $('#modelDialogBody').scrollTop = 0;
    }
  } else if (el.dataset.view) {
    event.preventDefault();
    if (el.id === 'labLibraryLink' && state.catalog) {
      state.libraryStage = 'explore';
      $('#libraryTask').value = state.labFilter;
      $('#libraryMode').value = '';
      renderLibrary();
      $('#librarySections').scrollTop = 0;
    }
    go(el.dataset.view);
  } else if (el.dataset.try) openTrial(el.dataset.try);
  else if (el.dataset.download) download(el.dataset.download, el.dataset.dmode || 'local');
});
window.addEventListener('hashchange', showView);

$('#modelDialog').addEventListener('close', event => {
  const id = event.currentTarget.dataset.modelId;
  const trigger = $('[data-model-details="' + CSS.escape(id) + '"]');
  (trigger || $('#librarySections')).focus({ preventScroll: true });
});

for (const button of $$('[data-library-stage]')) {
  button.addEventListener('click', () => {
    state.libraryStage = button.dataset.libraryStage;
    $('#librarySections').scrollTop = 0;
    renderLibrary();
  });
}
for (const id of ['libraryTask', 'libraryMode']) {
  $('#' + id).addEventListener('change', renderLibrary);
}
$('#librarySections').addEventListener('change', event => {
  const id = event.target.dataset.selectModel;
  if (!id) return;
  if (event.target.checked) state.selected.set(id, selectionMode(state.catalog.candidates.find(c => c.id === id)));
  else state.selected.delete(id);
  renderLibrary();
  const checkbox = $('[data-select-model="' + CSS.escape(id) + '"]');
  checkbox?.focus({ preventScroll: true });
});
$('#selectVisible').addEventListener('click', () => {
  const clear = state.visibleCandidates.every(c => state.selected.has(c.id));
  for (const c of state.visibleCandidates) {
    if (clear) state.selected.delete(c.id);
    else if (!state.selected.has(c.id)) state.selected.set(c.id, selectionMode(c));
  }
  renderLibrary();
});
$('#downloadSelected').addEventListener('click', downloadSelection);

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

$('#labTasks').addEventListener('change', event => {
  if (state.chat || state.mediaRunning) return;
  state.labFilter = event.target.value;
  renderLab();
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
    $('.connection').dataset.state = 'error';
    if (!state.catalog) say('libraryStatus', '正在等待本机服务');
  }
}
showView(false);
connect();
setInterval(connect, 4500);
