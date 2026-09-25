let sort = { by: 'track_start_dt_tm', dir: 'desc' };
const LEAD_IN_S = 2;
const cat = document.getElementById('filterCatId');
const after = document.getElementById('filterTrackTimeAfter');
const before = document.getElementById('filterTrackTimeBefore');
const tbody = document.getElementById('tracksBody');
const video = document.getElementById('videoPlayer');
const videoWrapper = document.querySelector('.video-wrapper');
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const metadataVideoName = document.getElementById('metadataVideoName');
const metadataTrackStart = document.getElementById('metadataTrackStart');
const metadataTrackStop = document.getElementById('metadataTrackStop');
let annotationData = null;

document.addEventListener('DOMContentLoaded', () => {
    cat.addEventListener('change', update);
    after.addEventListener('change', update);
    before.addEventListener('change', update);
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            sort.dir = sort.by === th.dataset.field && sort.dir === 'asc' ? 'desc' : 'asc';
            sort.by = th.dataset.field;
            update();
        });
    });
    video.addEventListener('loadedmetadata', () => {
        overlayCanvas.width = video.videoWidth;
        overlayCanvas.height = video.videoHeight;
        videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    });
    startOverlayLoop();
    setInterval(update, 5000);
    update();
});

async function update() {
    const p = new URLSearchParams({
        sort_by: sort.by,
        sort_dir: sort.dir,
        filter_cat_id: cat.value,
        filter_track_time_after: after.value,
        filter_track_time_before: before.value,
    });
    const tracks = await fetch(`/api/tracks?${p}`).then(r => r.json());
    
    const cats = [...new Set(tracks.map(t => t.cat_id))];
    const c = cat.value;
    cat.innerHTML = '<option value="">All Cats</option>';
    cats.forEach(x => {
        const o = document.createElement('option');
        o.value = x;
        o.textContent = x;
        cat.appendChild(o);
    });
    cat.value = c;
    
    tbody.innerHTML = '';
    tracks.forEach(t => {
        const row = tbody.insertRow();
        row.innerHTML = `<td>${new Date(t.track_start_dt_tm).toLocaleString()}</td><td>${t.cat_id}</td><td>${formatDuration(t.duration_s)}</td>`;
        if (!t.files_ready) row.classList.add('not-ready');
        row.onclick = () => {
            document.querySelectorAll('tbody tr').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
            metadataVideoName.textContent = t.video_name;
            metadataTrackStart.textContent = formatTrackTime(t.track_elapsed_start_s);
            metadataTrackStop.textContent = formatTrackTime(t.track_elapsed_end_s);
            video.src = `/video/${t.video_name}`;
            video.currentTime = Math.max(0, t.track_elapsed_start_s - LEAD_IN_S);
            video.play();
            annotationData = null;
            fetch(`/api/tracks/${t.manager_id}/${t.track_id}/annotations`)
                .then(r => r.json())
                .then(d => { annotationData = d; });
        };
    });
    
    document.querySelectorAll('th.sortable').forEach(th => {
        th.textContent = th.textContent.replace(/\s[↑↓]$/, '');
        if (th.dataset.field === sort.by) th.textContent += ` ${sort.dir === 'asc' ? '↑' : '↓'}`;
    });
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

function toPixel([xc, yc, w, h]) {
    const cw = overlayCanvas.width, ch = overlayCanvas.height;
    return {
        x1: (xc - w / 2) * cw,
        y1: (yc - h / 2) * ch,
        x2: (xc + w / 2) * cw,
        y2: (yc + h / 2) * ch,
        cx: xc * cw,
        cy: yc * ch,
    };
}

function drawOverlay(currentTime) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!annotationData || !annotationData.fps) return;

    const { fps, history_duration_s, frames, colour } = annotationData;
    const index = Math.round(currentTime * fps);
    const historyFrames = Math.round(history_duration_s * fps);
    const strokeColour = `rgb(${colour.join(',')})`;
    const dim = Math.min(overlayCanvas.width, overlayCanvas.height);
    const trailThickness = dim / 500;
    const boxThickness = dim / 250;

    // history trail
    overlayCtx.strokeStyle = strokeColour;
    overlayCtx.fillStyle = strokeColour;
    overlayCtx.lineWidth = trailThickness;
    let prevPoint = null;
    for (let i = Math.max(0, index - historyFrames); i <= index; i++) {
        const bbox = frames[i];
        if (!bbox) {
            prevPoint = null;
            continue;
        }
        const { cx, cy } = toPixel(bbox);
        overlayCtx.beginPath();
        overlayCtx.arc(cx, cy, trailThickness * 2, 0, 2 * Math.PI);
        overlayCtx.fill();
        if (prevPoint) {
            overlayCtx.beginPath();
            overlayCtx.moveTo(prevPoint.cx, prevPoint.cy);
            overlayCtx.lineTo(cx, cy);
            overlayCtx.stroke();
        }
        prevPoint = { cx, cy };
    }

    // current bounding box
    const currentBbox = frames[index];
    if (currentBbox) {
        const { x1, y1, x2, y2 } = toPixel(currentBbox);
        overlayCtx.lineWidth = boxThickness;
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
}

function startOverlayLoop() {
    if (video.requestVideoFrameCallback) {
        const cb = (now, metadata) => {
            drawOverlay(metadata.mediaTime);
            video.requestVideoFrameCallback(cb);
        };
        video.requestVideoFrameCallback(cb);
    } else {
        const raf = () => {
            drawOverlay(video.currentTime);
            requestAnimationFrame(raf);
        };
        raf();
    }
}

