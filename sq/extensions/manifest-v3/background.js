// ============================================================================
// SYNAPSE-QUANTUM: Manifest V3 Background Service Worker (Nodes 9, 10, 14)
// - Node 14: tab discarder. Skips pinned / audible / active tabs and the app itself.
// - Node 10: grant scraper. Queries the public Grants.gov search API. Demo placeholders are
//            clearly marked DEMO; scraped rows carry no invented stipend.
// - Node 9 : media gate state (unlock timestamp in chrome.storage.local).
//   Note: the gate is a self-discipline tool, not a security boundary.
// ============================================================================

const APP_HOSTS = ['localhost:5173', '127.0.0.1:5173'];

const DEMO_KANBAN = [
  { id: 'DEMO-01', title: 'DEMO: Cortical organoid reservoir computing grant', agency: 'Placeholder', stipend: 'n/a', stage: 'DEMO' },
  { id: 'DEMO-02', title: 'DEMO: Stochastic volatility PDE solver fellowship', agency: 'Placeholder', stipend: 'n/a', stage: 'DEMO' },
  { id: 'DEMO-03', title: 'DEMO: Two-photon holography fellowship', agency: 'Placeholder', stipend: 'n/a', stage: 'DEMO' },
];

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ mediaGateUnlockedUntil: 0, discardedTabsTotal: 0, kanbanGrants: DEMO_KANBAN });
});

async function searchGrants(keyword) {
  const res = await fetch('https://api.grants.gov/v1/api/search2', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keyword, oppStatuses: 'forecasted|posted', rows: 10 }),
  });
  if (!res.ok) throw new Error(`Grants.gov HTTP ${res.status}`);
  const json = await res.json();
  const hits = json?.data?.oppHits ?? json?.oppHits ?? [];
  return hits.map((h) => ({
    id: String(h.number ?? h.id ?? crypto.randomUUID()),
    title: String(h.title ?? 'Untitled opportunity'),
    agency: String(h.agency ?? h.agencyCode ?? 'unknown agency'),
    stipend: 'see listing',
    stage: 'Scraped',
    url: h.id ? `https://www.grants.gov/search-results-detail/${h.id}` : undefined,
  }));
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'RECLAIM_WGSL_HEAP') {
    chrome.tabs.query({ active: false, discarded: false, pinned: false, audible: false }, (tabs) => {
      let count = 0;
      for (const tab of tabs) {
        const isApp = APP_HOSTS.some((h) => tab.url?.includes(h));
        if (tab.id && !isApp) { chrome.tabs.discard(tab.id); count++; }
      }
      chrome.storage.local.get(['discardedTabsTotal'], (res) => {
        const total = (res.discardedTabsTotal || 0) + count;
        chrome.storage.local.set({ discardedTabsTotal: total });
        sendResponse({ discardedCount: count, totalReclaimed: total });
      });
    });
    return true;
  }

  if (message.type === 'SCRAPE_QUANT_GRANTS') {
    const keyword = message.query || 'computational neuroscience';
    searchGrants(keyword)
      .then((items) => chrome.storage.local.get(['kanbanGrants'], (res) => {
        const existing = Array.isArray(res.kanbanGrants) ? res.kanbanGrants.filter((g) => g.stage !== 'DEMO') : [];
        const seen = new Set(existing.map((g) => g.id));
        const merged = [...items.filter((i) => !seen.has(i.id)), ...existing].slice(0, 50);
        chrome.storage.local.set({ kanbanGrants: merged });
        sendResponse({ status: 'OK', grants: merged });
      }))
      .catch((err) => sendResponse({ status: 'ERROR', error: String(err), grants: DEMO_KANBAN }));
    return true;
  }

  if (message.type === 'LOCK_MEDIA_GATE') {
    chrome.storage.local.set({ mediaGateUnlockedUntil: 0 }, () => sendResponse({ locked: true }));
    return true;
  }
});
