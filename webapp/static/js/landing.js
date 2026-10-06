/* Landing page: headline word reveal and the live class ladder demo. */
(function () {
  'use strict';

  /* Split the headline into words for the blur-in sequence */
  document.querySelectorAll('[data-split]').forEach((el) => {
    const words = el.textContent.trim().split(/\s+/);
    el.setAttribute('aria-label', el.textContent.trim());
    el.innerHTML = words.map((w, i) =>
      '<span class="w" aria-hidden="true" style="--i:' + i + '">' + w + '</span>').join(' ');
  });

  /* A photo that fails to load (for example with no internet) is removed
     so the layout closes up instead of showing a broken picture. */
  const mosaic = document.getElementById('mosaic');
  if (mosaic) {
    const drop = (img) => {
      const fig = img.closest('.shot');
      if (fig) fig.remove();
      const left = mosaic.querySelectorAll('.shot').length;
      mosaic.dataset.photos = left;
      if (!left) document.getElementById('campus').remove();
    };
    mosaic.querySelectorAll('img').forEach((img) => {
      if (img.complete && img.naturalWidth === 0) drop(img);
      else img.addEventListener('error', () => drop(img), { once: true });
    });
  }

  const demo = document.getElementById('demo');
  if (!demo) return;
  const rungs = {};
  demo.querySelectorAll('.rung').forEach((r) => { rungs[r.dataset.class] = r; });
  const sliders = demo.querySelectorAll('input[type="range"]');
  const live = document.getElementById('demo-live');
  const voteRf = document.getElementById('vote-rf');
  const voteGb = document.getElementById('vote-gb');
  let timer = null, seq = 0;

  function render(data) {
    data.proba.forEach((p, i) => {
      const rung = rungs[i];
      rung.style.setProperty('--p', p.toFixed(3));
      rung.querySelector('.rung-value').textContent = Math.round(p * 100) + '%';
      rung.classList.toggle('is-top', i === data.index);
    });
    voteRf.textContent = data.votes.random_forest;
    voteGb.textContent = data.votes.gradient_boosting;
  }

  function refresh() {
    const body = {};
    sliders.forEach((s) => { body[s.name] = Number(s.value); });
    const mine = ++seq;
    fetch(demo.dataset.endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((res) => res.ok ? res.json() : Promise.reject(res))
      .then((data) => { if (mine === seq) { render(data); live.textContent = 'Live model'; } })
      .catch(() => { live.textContent = 'Model offline'; });
  }

  sliders.forEach((s) => s.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(refresh, 90);
  }));
  refresh();
})();
