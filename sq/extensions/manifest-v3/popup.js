// Renders with textContent only: scraped titles come from the network and must never reach innerHTML.
function renderKanban(grants) {
  const container = document.getElementById('kanban-list');
  if (!container || !Array.isArray(grants)) return;
  container.textContent = '';
  grants.slice(0, 5).forEach((g) => {
    const row = document.createElement('div');
    row.style.marginBottom = '4px';
    row.textContent = `• [${g.stage}] ${g.title} (${g.stipend})`;
    container.appendChild(row);
  });
}

document.getElementById('btn-reclaim')?.addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'RECLAIM_WGSL_HEAP' }, (res) => {
    const el = document.getElementById('heap-status');
    if (el && res) el.textContent = `Discarded ${res.discardedCount} tabs (Total reclaimed: ${res.totalReclaimed})`;
  });
});

document.getElementById('btn-lock')?.addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'LOCK_MEDIA_GATE' }, () => {
    const el = document.getElementById('gate-status');
    if (el) el.textContent = 'Media Gate Locked. Quiz required on YouTube.';
  });
});

document.getElementById('btn-scrape')?.addEventListener('click', () => {
  const status = document.getElementById('scrape-status');
  if (status) status.textContent = 'Querying Grants.gov...';
  chrome.runtime.sendMessage({ type: 'SCRAPE_QUANT_GRANTS' }, (res) => {
    if (status) status.textContent = res?.status === 'OK' ? 'Live results from Grants.gov' : `Failed (${res?.error ?? 'no response'}); showing DEMO rows`;
    if (res?.grants) renderKanban(res.grants);
  });
});

chrome.storage.local.get(['kanbanGrants'], (res) => { if (res.kanbanGrants) renderKanban(res.kanbanGrants); });
