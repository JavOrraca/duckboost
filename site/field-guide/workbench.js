(() => {
  function initializeWorkbench(root) {
    const tools = root.querySelector('.workbench-tools');
    if (!tools.hidden) return;
    const tabs = [...tools.querySelectorAll('[role="tab"]')];
    const panels = tabs.map(tab => root.querySelector('#' + tab.getAttribute('aria-controls')));
    const copy = tools.querySelector('.workbench-copy');
    const label = copy.querySelector('span');
    const status = root.querySelector('.workbench-status');
    let active = 0;
    let copyRequest = 0;
    let resetLabelTimer;

    function resetCopy() {
      copyRequest++;
      clearTimeout(resetLabelTimer);
      label.textContent = 'Copy';
      status.textContent = '';
      status.removeAttribute('data-silent');
    }

    function activate(index) {
      resetCopy();
      active = index;
      tabs.forEach((tab, i) => {
        const selected = i === active;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
        panels[i].hidden = !selected;
      });
    }

    tabs.forEach((tab, index) => {
      panels[index].setAttribute('role', 'tabpanel');
      panels[index].setAttribute('aria-labelledby', tab.id);
      panels[index].tabIndex = 0;
      tab.addEventListener('click', () => activate(index));
      tab.addEventListener('keydown', event => {
        const next = {
          ArrowLeft: (index + tabs.length - 1) % tabs.length,
          ArrowRight: (index + 1) % tabs.length,
          Home: 0,
          End: tabs.length - 1
        }[event.key];
        if (typeof next !== 'number') return;
        event.preventDefault();
        activate(next);
        tabs[next].focus();
      });
    });

    copy.addEventListener('click', async () => {
      resetCopy();
      const request = copyRequest;
      const panel = panels[active];
      const code = panel.querySelector('pre code');
      try {
        await navigator.clipboard.writeText(code.textContent);
        if (request !== copyRequest) return;
        label.textContent = 'Copied';
        status.setAttribute('data-silent', '');
        status.textContent = 'SQL copied to clipboard.';
        resetLabelTimer = setTimeout(resetCopy, 2000);
      } catch {
        if (request !== copyRequest) return;
        panel.focus({ preventScroll: true });
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'SQL selected. Use your device’s Copy command.';
      }
    });

    activate(0);
    tools.hidden = false;
  }

  function initialize() {
    document.querySelectorAll('.sql-workbench').forEach(initializeWorkbench);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
