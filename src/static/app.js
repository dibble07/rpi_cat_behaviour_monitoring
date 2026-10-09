let sort = { by: 'track_start_dt_tm', dir: 'desc' };
const LEAD_IN_S = 2;
const cat = document.getElementById('filterCatId');
const behaviour = document.getElementById('filterBehaviour');
const behaviourOptions = document.getElementById('behaviourOptions');
const behaviourFilterSummary = document.getElementById('behaviourFilterSummary');
const after = document.getElementById('filterTrackTimeAfter');
const before = document.getElementById('filterTrackTimeBefore');
const trainingFilter = document.getElementById('filterTrainingVideo');
const tbody = document.getElementById('tracksBody');
const video = document.getElementById('videoPlayer');
const videoWrapper = document.querySelector('.video-wrapper');
const metadataVideoName = document.getElementById('metadataVideoName');
const metadataTrackStart = document.getElementById('metadataTrackStart');
const metadataTrackStop = document.getElementById('metadataTrackStop');
const trainingVideoCheckbox = document.getElementById('trainingVideoCheckbox');
const clearTrainingVideos = document.getElementById('clearTrainingVideos');
let currentVideoName = null;

document.addEventListener('DOMContentLoaded', () => {
    cat.addEventListener('change', update);
    behaviourOptions.addEventListener('change', update);
    document.addEventListener('click', event => {
        if (!behaviour.contains(event.target)) behaviour.open = false;
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && behaviour.open) {
            behaviour.open = false;
            behaviour.querySelector('summary').focus();
        }
    });
    after.addEventListener('change', update);
    before.addEventListener('change', update);
    trainingFilter.addEventListener('change', update);
    trainingVideoCheckbox.addEventListener('change', updateTrainingVideo);
    clearTrainingVideos.addEventListener('click', clearAllTrainingVideos);
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            sort.dir = sort.by === th.dataset.field && sort.dir === 'asc' ? 'desc' : 'asc';
            sort.by = th.dataset.field;
            update();
        });
    });
    video.addEventListener('loadedmetadata', () => {
        videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    });
    setInterval(update, 5000);
    update();
});

async function update() {
    const selected = behaviourOptions.querySelectorAll('input:checked');
    behaviourFilterSummary.textContent = selected.length ? `${selected.length} selected` : 'All Behaviours';
    const p = new URLSearchParams({
        sort_by: sort.by,
        sort_dir: sort.dir,
        filter_cat_id: cat.value,
        filter_track_time_after: after.value,
        filter_track_time_before: before.value,
    });
    if (trainingFilter.value) p.set('filter_training_video', trainingFilter.value);
    selected.forEach(checkbox => p.append('filter_behaviour', checkbox.value));
    const { tracks, filters, training_videos: trainingVideos } = await fetch(`/api/tracks?${p}`).then(r => r.json());
    if (currentVideoName) trainingVideoCheckbox.checked = trainingVideos.includes(currentVideoName);
    const selectedCat = cat.value;
    cat.replaceChildren(new Option('All Cats', ''), ...filters.cats.map(value => new Option(value, value)));
    cat.value = selectedCat;
    updateBehaviourOptions(filters.behaviours);
    
    tbody.innerHTML = '';
    tracks.forEach(t => {
        const row = tbody.insertRow();
        [new Date(t.track_start_dt_tm).toLocaleString(), t.cat_id,
            t.behaviours.join(', ') || '\u2014', formatDuration(t.duration_s)]
            .forEach(value => { row.insertCell().textContent = value; });
        if (!t.files_ready) row.classList.add('not-ready');
        row.onclick = () => {
            document.querySelectorAll('tbody tr').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
            metadataVideoName.textContent = t.video_name;
            metadataTrackStart.textContent = formatTrackTime(t.track_elapsed_start_s);
            metadataTrackStop.textContent = formatTrackTime(t.track_elapsed_end_s);
            currentVideoName = t.video_name;
            trainingVideoCheckbox.disabled = false;
            trainingVideoCheckbox.checked = t.is_training_video;
            if (!t.files_ready) {
                video.pause();
                video.removeAttribute('src');
                video.load();
                return;
            }
            video.src = `/video/${t.video_name}`;
            video.currentTime = Math.max(0, t.track_elapsed_start_s - LEAD_IN_S);
            video.play();
        };
    });
    
    document.querySelectorAll('th.sortable').forEach(th => {
        th.textContent = th.textContent.replace(/\s[↑↓]$/, '');
        if (th.dataset.field === sort.by) th.textContent += ` ${sort.dir === 'asc' ? '↑' : '↓'}`;
    });
}

async function updateTrainingVideo() {
    if (!currentVideoName) return;
    try {
        await fetchJson('/api/training-videos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_name: currentVideoName,
                is_training_video: trainingVideoCheckbox.checked,
            }),
        });
        await update();
    } catch (error) {
        console.error('Unable to update training video selection', error);
        alert('Unable to update training video selection. See the console for details.');
        trainingVideoCheckbox.checked = !trainingVideoCheckbox.checked;
    }
}

async function clearAllTrainingVideos() {
    if (!confirm('Clear all training video selections?')) return;
    try {
        await fetchJson('/api/training-videos', { method: 'DELETE' });
        await update();
    } catch (error) {
        console.error('Unable to clear training video selections', error);
        alert('Unable to clear training video selections. See the console for details.');
    }
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

function updateBehaviourOptions(values) {
    const existing = behaviourOptions.querySelectorAll('input');
    if (existing.length === values.length && values.every((value, index) => value === existing[index].value)) return;
    const selected = Array.from(existing).filter(checkbox => checkbox.checked).map(checkbox => checkbox.value);
    behaviourOptions.replaceChildren(...values.map(value => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = value;
        checkbox.checked = selected.includes(value);
        const text = document.createElement('span');
        text.textContent = value;
        label.append(checkbox, text);
        return label;
    }));
}

function formatDuration(seconds) {
    const totalSeconds = Math.floor(seconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const remainingSeconds = totalSeconds % 60;
    return [hours, minutes, remainingSeconds]
        .map(value => String(value).padStart(2, '0'))
        .join(':');
}

function formatTrackTime(seconds) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '\u2014';
    return formatDuration(seconds);
}
