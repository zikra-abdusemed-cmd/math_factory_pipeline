const form = document.querySelector('#generate-form');
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
const olxForm = document.querySelector('#olx-form');
const olxFile = document.querySelector('#olx-file');
const olxPreview = document.querySelector('#olx-preview');
const olxError = document.querySelector('#olx-error');
const unitSelect = document.querySelector('#unit-select');
const unitList = document.querySelector('#unit-list');
const scopeSummary = document.querySelector('#scope-summary');
const bundleResult = document.querySelector('#bundle-result');

let activeImport = null;
let activeManifest = null;

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
  [statusPanel, resultPanel, bundleResult, errorPanel].forEach((item) => {
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
  prompt.disabled = false;
  qualityOptions.querySelectorAll('input').forEach((input) => { input.disabled = false; });
  document.querySelector('#generate-course').disabled = false;
  document.querySelector('#preview-button').disabled = false;
  olxFile.disabled = false;
  unitSelect.querySelectorAll('input').forEach((input) => { input.disabled = false; });
}

function lockForm() {
  button.disabled = true;
  prompt.disabled = true;
  qualityOptions.querySelectorAll('input').forEach((input) => { input.disabled = true; });
  document.querySelector('#generate-course').disabled = true;
  document.querySelector('#preview-button').disabled = true;
  olxFile.disabled = true;
  unitSelect.querySelectorAll('input').forEach((input) => { input.disabled = true; });
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
  if (detail.unit_total) {
    statusText.textContent = `Unit ${detail.unit_index || 1} of ${detail.unit_total}: ${detail.unit_title || 'Preparing video'} — ${statusText.textContent}`;
  }

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
  if (detail.unit_total) bits.push(`course unit ${detail.unit_index || 1}/${detail.unit_total}`);
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
      if (job.download_name?.endsWith('.zip')) {
        document.querySelector('#bundle-download').href = job.video_url;
        document.querySelector('#bundle-summary').textContent = `${job.detail?.unit_total || 0} unit videos, with course order and titles in the manifest.`;
        show(bundleResult);
      } else {
        video.src = job.video_url;
        download.href = job.video_url;
        show(resultPanel);
      }
      unlockForm();
      return;
    }
    if (job.status === 'error') {
      stopElapsed();
      errorText.textContent = job.partial_url
        ? `${job.error || 'Course generation failed.'} Completed unit videos are available in a partial ZIP.`
        : (job.error || 'Video generation failed.');
      if (job.partial_url) {
        let partial = document.querySelector('#partial-download');
        if (!partial) {
          partial = document.createElement('a');
          partial.id = 'partial-download';
          partial.className = 'download';
          partial.textContent = 'Download completed unit videos';
          partial.download = true;
          errorPanel.append(partial);
        }
        partial.href = job.partial_url;
      }
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
  prompt.focus();
}

function fillSelect(select, entries) {
  select.innerHTML = entries.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)}</option>`).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

function selectedScopeUnits() {
  if (!activeManifest) return [];
  return [...unitSelect.querySelectorAll('.lesson-choice-input:checked')].flatMap((lessonInput) => {
    const section = activeManifest.sections.find((item) => item.id === lessonInput.dataset.sectionId);
    const subsection = section?.subsections.find((item) => item.id === lessonInput.dataset.subsectionId);
    const unit = subsection?.units.find((item) => item.id === lessonInput.value);
    return section && subsection && unit ? [{ section, subsection, unit, components: unit.components }] : [];
  });
}

function renderOlxPreview() {
  unitSelect.innerHTML = activeManifest.sections.map((section) => `
    <div class="unit-selection-group">
      <label class="unit-choice"><input class="unit-choice-input" type="checkbox" data-section-id="${escapeHtml(section.id)}"><span>${escapeHtml(section.title)}</span></label>
      ${section.subsections.map((subsection) => `<div class="subsection-group"><strong>${escapeHtml(subsection.title)}</strong>
        ${subsection.units.map((unit) => `<label class="content-choice"><input class="lesson-choice-input" type="checkbox" data-section-id="${escapeHtml(section.id)}" data-subsection-id="${escapeHtml(subsection.id)}" value="${escapeHtml(unit.id)}" disabled><span>${escapeHtml(unit.title)}${unit.components.length ? ` <small>(${unit.components.map((component) => escapeHtml(component.title)).join(', ')})</small>` : ''}</span></label>`).join('')}
      </div>`).join('')}
    </div>
  `).join('');
  unitSelect.onchange = (event) => {
    const sectionInput = event.target.closest('.unit-choice-input');
    if (sectionInput) {
      unitSelect.querySelectorAll(`.lesson-choice-input[data-section-id="${sectionInput.dataset.sectionId}"]`).forEach((input) => {
        input.disabled = !sectionInput.checked;
        input.checked = sectionInput.checked;
      });
    } else {
      const lessonInput = event.target.closest('.lesson-choice-input');
      if (lessonInput) {
        const parent = unitSelect.querySelector(`.unit-choice-input[data-section-id="${lessonInput.dataset.sectionId}"]`);
        const children = [...unitSelect.querySelectorAll(`.lesson-choice-input[data-section-id="${lessonInput.dataset.sectionId}"]`)];
        parent.checked = children.some((input) => input.checked);
        parent.indeterminate = children.some((input) => input.checked) && children.some((input) => !input.checked);
      }
    }
    updateScopePreview();
  };
  document.querySelector('#toggle-subsections').onclick = () => {
    const boxes = [...unitSelect.querySelectorAll('.lesson-choice-input:not(:disabled)')];
    const selectAll = boxes.some((input) => !input.checked);
    boxes.forEach((input) => { input.checked = selectAll; });
    unitSelect.querySelectorAll('.unit-choice-input').forEach((parent) => {
      const children = [...unitSelect.querySelectorAll(`.lesson-choice-input[data-section-id="${parent.dataset.sectionId}"]`)];
      parent.checked = children.some((input) => input.checked);
      parent.indeterminate = children.some((input) => input.checked) && children.some((input) => !input.checked);
    });
    updateScopePreview();
  };
  updateScopePreview();
  document.querySelector('#course-title').textContent = `${activeManifest.title} · Choose units, then content parts`;
  olxPreview.hidden = false;
}

function updateScopePreview() {
  const selected = selectedScopeUnits();
  const enabledContent = [...unitSelect.querySelectorAll('.lesson-choice-input:not(:disabled)')];
  document.querySelector('#toggle-subsections').textContent = enabledContent.length && enabledContent.every((input) => input.checked)
    ? 'Clear selected lesson subsections'
    : 'Select all lesson subsections in chosen units';
  document.querySelector('#toggle-subsections').disabled = enabledContent.length === 0;
  scopeSummary.textContent = `${selected.length} video${selected.length === 1 ? '' : 's'} will be generated from ${selected.length} selected unit${selected.length === 1 ? '' : 's'}.`;
  unitList.innerHTML = selected.map(({ section, subsection, unit, components }) => `<li><strong>${escapeHtml(section.title)} / ${escapeHtml(subsection.title)} / ${escapeHtml(unit.title)}</strong><ul>${components.map((component) => `<li>${escapeHtml(component.title)} · ${escapeHtml(component.kind)} (${component.characters} characters)</li>`).join('')}</ul></li>`).join('');
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
document.querySelector('#bundle-new-button').addEventListener('click', resetToForm);

olxForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!olxFile.files?.length) return;
  olxError.hidden = true;
  olxPreview.hidden = true;
  document.querySelector('#preview-button').disabled = true;
  try {
    const selectedFiles = [...olxFile.files];
    const relevantFiles = selectedFiles.filter((file) => /(^|\/)(course\.xml|course\/[^/]+\.xml|chapter\/[^/]+\.xml|sequential\/[^/]+\.xml|vertical\/[^/]+\.xml|html\/[^/]+\.(xml|html)|problem\/[^/]+\.xml|video\/[^/]+\.xml|static\/[^/]+\.htm|static\/html5\/.*\/index\.html)$/i.test(file.webkitRelativePath));
    const files = await Promise.all(relevantFiles.map(async (file) => ({
      path: file.webkitRelativePath.split('/').slice(1).join('/'),
      content: await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(',')[1]);
        reader.onerror = () => reject(reader.error || new Error(`Could not read ${file.name}`));
        reader.readAsDataURL(file);
      }),
    })));
    const response = await fetch('/api/olx/preview-folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not preview this OLX course.');
    activeImport = body.import_id;
    activeManifest = body.manifest;
    renderOlxPreview();
  } catch (error) {
    olxError.textContent = error.message || 'Could not preview this OLX course.';
    olxError.hidden = false;
  } finally {
    document.querySelector('#preview-button').disabled = false;
  }
});

document.querySelector('#generate-course').addEventListener('click', async () => {
  if (!activeImport) return;
  const selections = selectedScopeUnits().map(({ section, subsection, unit, components }) => ({
    section_id: section.id,
    subsection_id: subsection.id,
    unit_id: unit.id,
    component_indices: components.map((component) => Number(component.id)),
  }));
  if (!selections.length) {
    olxError.textContent = 'Check at least one section/unit and at least one content subsection inside it.';
    olxError.hidden = false;
    return;
  }
  const request = {
    import_id: activeImport,
    selections,
    prompt: prompt.value || null,
    quality: selectedQuality,
  };
  lockForm();
  progressBar.style.width = '4%';
  statusText.textContent = 'Preparing selected course videos…';
  statusDetail.textContent = scopeSummary.textContent;
  renderStageList('starting');
  show(statusPanel);
  startElapsed();
  try {
    const response = await fetch('/api/olx/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Could not start course generation.');
    poll(body.job_id);
  } catch (error) {
    stopElapsed();
    errorText.textContent = error.message || 'Could not start course generation.';
    show(errorPanel);
    unlockForm();
  }
});

loadConfig();
