const form = document.querySelector('#generate-form');
const title = document.querySelector('#title');
const prompt = document.querySelector('#prompt');
const button = document.querySelector('#generate-button');
const statusPanel = document.querySelector('#status-panel');
const resultPanel = document.querySelector('#result-panel');
const errorPanel = document.querySelector('#error-panel');
const statusText = document.querySelector('#status-text');
const statusDetail = document.querySelector('#status-detail');
const progressBar = document.querySelector('#progress-bar');
const errorText = document.querySelector('#error-text');
const video = document.querySelector('#video');
const download = document.querySelector('#download-link');
const elapsedEl = document.querySelector('#elapsed');
const stageList = document.querySelector('#stage-list');
const qualityOptions = document.querySelector('#quality-options');
const configHint = document.querySelector('#config-hint');

const STAGE_COPY = {
  starting: ['Preparing your job…', 4],
  generating_script: ['Writing the narration script…', 10],
  script_generated: ['Narration script ready.', 18],
  generating_storyboard: ['Planning the visual storyboard…', 24],
  storyboard_generated: ['Storyboard ready.', 30],
  generating_scene_code: ['Writing Manim code for scene {index} of {total}…', 34],
  scene_code_generated: ['Scene {index} of {total} validated.', 48],
  pregenerating_voiceovers: ['Pre-generating narration audio…', 52],
  voiceovers_ready: ['Narration audio cached.', 58],
  rendering_scene: ['Rendering scene {index} of {total}…', 62],
  scene_rendered: ['Rendered scene {index} of {total}.', 82],
  stitching: ['Stitching the final video…', 92],
  final_video_stitched: ['Finalizing your video…', 98],
  done: ['Your video is ready.', 100],
};

const PIPELINE_STEPS = [
  { id: 'script', label: 'Script', stages: ['generating_script', 'script_generated'] },
  { id: 'storyboard', label: 'Storyboard', stages: ['generating_storyboard', 'storyboard_generated'] },
  { id: 'codegen', label: 'Scene code', stages: ['generating_scene_code', 'scene_code_generated'] },
  { id: 'tts', label: 'Voice cache', stages: ['pregenerating_voiceovers', 'voiceovers_ready'] },
  { id: 'render', label: 'Render', stages: ['rendering_scene', 'scene_rendered'] },
  { id: 'stitch', label: 'Stitch', stages: ['stitching', 'final_video_stitched', 'done'] },
];

let selectedQuality = 'qm';
let startedAt = null;
let elapsedTimer = null;

function show(panel) {
  [statusPanel, resultPanel, errorPanel].forEach((item) => {
    item.hidden = item !== panel;
  });
}

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = String(total % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function startElapsed() {
  startedAt = Date.now();
  elapsedEl.textContent = '0:00';
  window.clearInterval(elapsedTimer);
  elapsedTimer = window.setInterval(() => {
    elapsedEl.textContent = formatElapsed(Date.now() - startedAt);
  }, 1000);
}

function stopElapsed() {
  window.clearInterval(elapsedTimer);
  elapsedTimer = null;
  if (startedAt) elapsedEl.textContent = formatElapsed(Date.now() - startedAt);
}

function unlockForm() {
  button.disabled = false;
  title.disabled = false;
  prompt.disabled = false;
  qualityOptions.querySelectorAll('input').forEach((input) => { input.disabled = false; });
}

function lockForm() {
  button.disabled = true;
  title.disabled = true;
  prompt.disabled = true;
  qualityOptions.querySelectorAll('input').forEach((input) => { input.disabled = true; });
}

function stepState(stage, step) {
  const order = PIPELINE_STEPS.map((item) => item.id);
  const current = PIPELINE_STEPS.find((item) => item.stages.includes(stage));
  if (!current) return stage === 'done' ? 'done' : '';
  const currentIndex = order.indexOf(current.id);
  const stepIndex = order.indexOf(step.id);
  if (stage === 'done' || stepIndex < currentIndex) return 'done';
  if (stepIndex === currentIndex) return 'active';
  return '';
}

function renderStageList(stage, detail = {}) {
  const index = detail.scene_index;
  const total = detail.scene_total;
  stageList.innerHTML = PIPELINE_STEPS.map((step) => {
    const state = stepState(stage, step);
    let label = step.label;
    if ((step.id === 'codegen' || step.id === 'render') && total) {
      label = `${step.label} (${index || 1}/${total})`;
    }
    if (step.id === 'tts' && detail.count != null) {
      label = `${step.label} (${detail.count})`;
    }
    return `<li class="${state}">${label}</li>`;
  }).join('');
}

function updateStatus(job) {
  const [template, baseline] = STAGE_COPY[job.stage] || ['Working on your video…', 10];
  const detail = job.detail || {};
  const index = detail.scene_index || 1;
  const total = detail.scene_total || 1;
  statusText.textContent = template.replace('{index}', index).replace('{total}', total);

  let progress = baseline;
  if (job.stage.includes('scene_code') && detail.scene_total) {
    progress = 30 + ((index - (job.stage === 'generating_scene_code' ? 1 : 0)) / total) * 18;
  }
  if (job.stage.includes('rendering') || job.stage === 'scene_rendered') {
    progress = 58 + ((index - (job.stage === 'rendering_scene' ? 1 : 0)) / total) * 30;
  }
  progressBar.style.width = `${Math.min(99, Math.max(baseline, progress))}%`;

  const bits = [];
  if (detail.class_name) bits.push(detail.class_name);
  if (detail.quality) bits.push(`quality ${detail.quality}`);
  if (detail.count != null) bits.push(`${detail.count} narrations`);
  statusDetail.textContent = bits.join(' · ');
  renderStageList(job.stage, detail);
}

async function poll(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('The local server restarted while this video was generating. Restart the job after starting Uvicorn without --reload or multiple workers.');
      }
      throw new Error(job.detail || 'Could not read job status.');
    }
    if (job.status === 'done') {
      progressBar.style.width = '100%';
      renderStageList('done', job.detail || {});
      stopElapsed();
      video.src = job.video_url;
      download.href = job.video_url;
      show(resultPanel);
      unlockForm();
      return;
    }
    if (job.status === 'error') {
      stopElapsed();
      errorText.textContent = job.error || 'Video generation failed.';
      show(errorPanel);
      unlockForm();
      return;
    }
    updateStatus(job);
    window.setTimeout(() => poll(jobId), 1200);
  } catch (error) {
    stopElapsed();
    errorText.textContent = error.message || 'Could not check the job status.';
    show(errorPanel);
    unlockForm();
  }
}

function resetToForm() {
  stopElapsed();
  unlockForm();
  video.removeAttribute('src');
  video.load();
  show(null);
  title.focus();
}

function renderQualityOptions(qualities, defaultQuality) {
  selectedQuality = defaultQuality || 'qm';
  qualityOptions.innerHTML = qualities.map((item) => `
    <label>
      <input type="radio" name="quality" value="${item.id}" ${item.id === selectedQuality ? 'checked' : ''}>
      <span>
        <span class="q-label">${item.label}</span>
        <span class="q-hint">${item.hint}</span>
      </span>
    </label>
  `).join('');

  qualityOptions.querySelectorAll('input').forEach((input) => {
    input.addEventListener('change', () => {
      if (input.checked) selectedQuality = input.value;
    });
  });
}

async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error('config unavailable');
    const data = await response.json();
    renderQualityOptions(data.qualities || [], data.default_quality || 'qm');
    const voice = data.voice_provider === 'offline' ? 'silent offline audio' : 'OpenAI TTS';
    configHint.textContent = `Using ${voice}. One job at a time.`;
  } catch {
    renderQualityOptions([
      { id: 'ql', label: 'Draft', hint: 'Fast, low resolution' },
      { id: 'qm', label: 'Balanced', hint: 'Good for most runs' },
      { id: 'qh', label: 'High', hint: '1080p — slower' },
      { id: 'qk', label: '4K', hint: 'Slowest, largest files' },
    ], 'qm');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  lockForm();
  progressBar.style.width = '4%';
  statusText.textContent = 'Starting your video…';
  statusDetail.textContent = `quality ${selectedQuality}`;
  renderStageList('starting');
  show(statusPanel);
  startElapsed();
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title.value,
        prompt: prompt.value || null,
        quality: selectedQuality,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not start video generation.');
    poll(body.job_id);
  } catch (error) {
    stopElapsed();
    errorText.textContent = error.message || 'Could not start video generation.';
    show(errorPanel);
    unlockForm();
  }
});

document.querySelector('#retry-button').addEventListener('click', resetToForm);
document.querySelector('#new-button').addEventListener('click', resetToForm);

loadConfig();
