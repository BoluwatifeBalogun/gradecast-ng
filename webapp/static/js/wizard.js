/* Prediction form. Works as one long form without JavaScript.
   Steps can be visited in any order once the student's name and
   institution type are filled in, and "Predict now" works on every step:
   anything not changed keeps the value of a typical student. */
(function () {
  'use strict';
  const form = document.getElementById('wizard');
  if (!form) return;
  const panels = Array.from(form.querySelectorAll('.panel'));
  const tabs = Array.from(form.querySelectorAll('.progress button'));
  const now = document.getElementById('step-now');
  let current = -1;
  let unlocked = form.dataset.open === '1';

  panels.forEach((p, i) => {
    if (p.querySelector('.has-error')) tabs[i].classList.add('has-error');
  });

  function show(index, back) {
    index = Math.max(0, Math.min(panels.length - 1, index));
    panels.forEach((p, i) => {
      p.classList.toggle('is-current', i === index);
      p.classList.toggle('back', !!back);
    });
    tabs.forEach((t, i) => {
      t.classList.toggle('is-current', i === index);
      t.classList.toggle('is-done', unlocked && i !== index);
      t.setAttribute('aria-selected', String(i === index));
    });
    now.textContent = index + 1;
    if (current !== -1) {
      const heading = panels[index].querySelector('h2');
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
      form.scrollIntoView({ behavior: GC.reduce ? 'auto' : 'smooth', block: 'start' });
    }
    current = index;
  }

  /* Messages next to a field, added and removed in place */
  function setError(input, message) {
    const field = input.closest('.field');
    const old = field.querySelector('.error-text');
    if (old) old.remove();
    if (!message) {
      field.classList.remove('has-error');
      input.removeAttribute('aria-invalid');
      return true;
    }
    field.classList.add('has-error');
    input.setAttribute('aria-invalid', 'true');
    const msg = document.createElement('span');
    msg.className = 'error-text';
    msg.id = 'err-' + input.id;
    msg.textContent = message;
    field.appendChild(msg);
    return false;
  }

  /* Step one holds the two required fields */
  function studentValid() {
    const name = document.getElementById('student_name');
    const inst = document.getElementById('institution_type');
    const okName = setError(name, name.value.trim().length >= 2 ? '' : "Enter the student's name.");
    const okInst = setError(inst, inst.value ? '' : 'Choose the institution type. It decides which classes are shown.');
    if (!okName) name.focus(); else if (!okInst) inst.focus();
    if (okName && okInst) { unlocked = true; tabs[0].classList.remove('has-error'); }
    return okName && okInst;
  }

  form.addEventListener('click', (e) => {
    const next = e.target.closest('[data-next]');
    const back = e.target.closest('[data-back]');
    const go = e.target.closest('[data-go]');
    if (next && (current !== 0 || studentValid())) show(current + 1);
    if (back) show(current - 1, true);
    if (go && !go.disabled) {
      const target = Number(go.dataset.go);
      if (target === 0 || current !== 0 || studentValid()) show(target, target < current);
    }
  });

  /* Enter in a text field moves on instead of submitting half a form */
  form.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.matches('input.input')) {
      e.preventDefault();
      if (current !== 0 || studentValid()) show(current + 1);
    }
  });
  form.addEventListener('submit', (e) => {
    if (!studentValid()) {
      e.preventDefault();
      form.querySelectorAll('.is-loading').forEach((b) => b.classList.remove('is-loading'));
      show(0, true);
      studentValid();
    }
  }, true);

  /* Live preview of the grading scale for the chosen institution type */
  const inst = document.getElementById('institution_type');
  const preview = document.getElementById('scale-preview');
  if (inst && preview) {
    const scales = JSON.parse(preview.dataset.scales);
    const map = JSON.parse(preview.dataset.map);
    inst.addEventListener('change', () => {
      const key = map[inst.value];
      if (!key) return;
      const s = scales[key];
      preview.textContent = 'Results will use ' + s.name.toLowerCase() + ' classes: ' +
        s.labels.slice().reverse().join(', ') + '.';
      if (inst.value) setError(inst, '');
    });
  }

  /* Typed number and slider stay in step */
  form.querySelectorAll('.num-input').forEach((box) => {
    const range = document.getElementById(box.dataset.for);
    const clamp = (v) => Math.min(+range.max, Math.max(+range.min, Math.round(v)));
    box.addEventListener('input', () => {
      if (box.value === '' || isNaN(+box.value)) return;
      range.value = clamp(+box.value);
      GC.paint(range);
    });
    box.addEventListener('blur', () => {
      box.value = isNaN(+box.value) || box.value === '' ? range.value : clamp(+box.value);
      range.value = box.value;
      GC.paint(range);
    });
    range.addEventListener('input', () => { box.value = range.value; });
  });

  show(Number(form.dataset.start) || 0);
})();
