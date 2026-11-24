const tabs = [...document.querySelectorAll('[role="tab"]')];
const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
const copy = document.querySelector('.copy');
const copyLabel = copy.querySelector('span');
let selectedIndex = 0;
let copyTimer;

function selectTab(index) {
  selectedIndex = index;
  tabs.forEach((tab, i) => {
    const selected = i === index;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    panels[i].hidden = !selected;
  });
  clearTimeout(copyTimer);
  copyLabel.textContent = 'Copy';
}

tabs.forEach((tab, index) => {
  panels[index].setAttribute('role', 'tabpanel');
  panels[index].setAttribute('aria-labelledby', tab.id);
  panels[index].tabIndex = 0;
  tab.addEventListener('click', () => selectTab(index));
  tab.addEventListener('keydown', event => {
    const positions = {
      ArrowLeft: (index + tabs.length - 1) % tabs.length,
      ArrowRight: (index + 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1
    };
    if (!Object.hasOwn(positions, event.key)) return;
    event.preventDefault();
    selectTab(positions[event.key]);
    tabs[selectedIndex].focus();
  });
});

copy.addEventListener('click', async () => {
  const code = panels[selectedIndex].querySelector('code');
  clearTimeout(copyTimer);
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyLabel.textContent = 'Copied';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    copyLabel.textContent = 'Selected';
  }
  copyTimer = setTimeout(() => { copyLabel.textContent = 'Copy'; }, 1800);
});

selectTab(0);
document.querySelector('.switcher').hidden = false;
