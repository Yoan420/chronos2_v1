/* The sandboxed preview requests read-only navigation from its owning NYX page. */
(() => {
  'use strict';
  function mount() {
    if (window.parent === window) return;
    const date = document.querySelector('time.delivery-date');
    if (!date) return;
    const initialDay = date.getAttribute('datetime') || date.textContent.trim();
    const select = document.createElement('select');
    select.className = 'delivery-date delivery-date-picker';
    select.setAttribute('aria-label', 'Date de livraison du rapport CWE');
    select.title = 'Afficher les résultats d’une autre livraison';
    select.disabled = true;
    const initial = document.createElement('option');
    initial.value = initialDay;
    initial.textContent = initialDay;
    select.append(initial);
    let mounted = false;
    window.addEventListener('message', event => {
      if (event.source !== window.parent || event.data?.type !== 'nyx:delivery-options') return;
      const dates = event.data.dates;
      if (!Array.isArray(dates) || !dates.length || dates.length > 10000 ||
          !dates.every(day => typeof day === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(day))) return;
      // A new list must never relabel an older report still open in a dialog.
      const selected = initialDay;
      select.replaceChildren(...[...new Set([...dates, initialDay])].map(day => {
        const option = document.createElement('option');
        option.value = day;
        option.textContent = day;
        return option;
      }));
      select.value = selected;
      select.disabled = false;
      if (!mounted) { date.replaceWith(select); mounted = true; }
      if (event.data.focus === true) select.focus();
    });
    select.addEventListener('change', () => {
      window.parent.postMessage({type:'nyx:delivery-selected', date:select.value}, '*');
    });
    window.parent.postMessage({type:'nyx:delivery-ready'}, '*');
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once:true});
  else mount();
})();
