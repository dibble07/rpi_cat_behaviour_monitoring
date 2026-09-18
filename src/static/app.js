let sort = { by: 'track_start_dt_tm', dir: 'desc' };
const cat = document.getElementById('filterCatId');
const after = document.getElementById('filterTrackTimeAfter');
const before = document.getElementById('filterTrackTimeBefore');
const tbody = document.getElementById('tracksBody');
const video = document.getElementById('videoPlayer');

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
        row.innerHTML = `<td>${t.video_name}</td><td>${t.cat_id}</td><td>${t.track_elapsed_start_s.toFixed(1)}</td><td>${t.track_elapsed_end_s.toFixed(1)}</td><td>${new Date(t.track_start_dt_tm).toLocaleString()}</td>`;
        row.onclick = () => {
            document.querySelectorAll('tbody tr').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
            video.src = `/video/${t.video_name}`;
            video.currentTime = t.track_elapsed_start_s;
            video.play();
        };
    });
    
    document.querySelectorAll('th.sortable').forEach(th => {
        th.textContent = th.textContent.replace(/\s[↑↓]$/, '');
        if (th.dataset.field === sort.by) th.textContent += ` ${sort.dir === 'asc' ? '↑' : '↓'}`;
    });
}
