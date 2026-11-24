(() => {
  const finder = document.querySelector('.reference-finder');
  if (!finder) return;
  const input = finder.querySelector('input');
  const status = finder.querySelector('[role="status"]');
  const directory = finder.querySelector('.reference-directory');
  const groups = [...finder.querySelectorAll('.reference-group')];
  const returnLink = document.querySelector('.reference-return');
  const compact = window.matchMedia('(max-width: 48rem)');
  let previousOpen = null;
  let directoryWasOpen;

  function sizeDirectory() {
    directory.open = !compact.matches || Boolean(input.value.trim());
  }
  compact.addEventListener('change', sizeDirectory);
  sizeDirectory();

  function normalize(value) {
    return value.toLowerCase().replace(/[_-]/g, ' ').trim();
  }

  const catalog = groups.map(group => ({
    group,
    entries: [...group.querySelectorAll('li')].map(item => ({
      item,
      link: item.querySelector('a'),
      terms: normalize(group.querySelector('summary').textContent + ' ' + item.textContent)
    }))
  }));

  function filter() {
    const query = normalize(input.value);
    const words = query ? query.split(/\s+/) : [];
    if (query && !previousOpen) {
      previousOpen = groups.map(group => group.open);
      directoryWasOpen = directory.open;
    }
    if (query) directory.open = true;
    else if (previousOpen) directory.open = !compact.matches || directoryWasOpen;
    let count = 0;
    catalog.forEach(({ group, entries }, index) => {
      let found = false;
      for (const { item, terms } of entries) {
        const match = words.every(word => terms.includes(word));
        item.hidden = !match;
        if (match) { found = true; count++; }
      }
      group.hidden = !found;
      if (query) group.open = found;
      else if (previousOpen) group.open = previousOpen[index];
    });
    if (!query) previousOpen = null;
    status.textContent = query ? (count ? count + (count === 1 ? ' function' : ' functions') : 'No matching functions.') : '';
  }

  function markLocation(preferredLink) {
    const links = catalog.flatMap(({ entries }) => entries.map(entry => entry.link));
    const current = preferredLink || links.find(link => link.hash === location.hash);
    for (const link of links) {
      if (link === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
    if (current) current.closest('.reference-group').open = true;
  }

  input.addEventListener('input', filter);
  finder.querySelector('.reference-filter').hidden = false;
  let requestedLink;
  finder.addEventListener('click', event => {
    const link = event.target.closest('.reference-group a');
    if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    requestedLink = link;
    markLocation(link);
  });
  window.addEventListener('hashchange', () => {
    markLocation(requestedLink?.hash === location.hash ? requestedLink : null);
    requestedLink = null;
  });
  markLocation();

  if (returnLink && 'IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      returnLink.hidden = entry.isIntersecting || entry.boundingClientRect.top >= 0;
    }).observe(finder);
    returnLink.addEventListener('click', () => {
      requestAnimationFrame(() => input.focus({ preventScroll: true }));
    });
  }
})();
