const BDT_PER_USD = 123.8377; // Bangladesh Bank reference rate, 26 Oct 2026.
const usd = new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});
const formatPrice = value => usd.format(Number(value) / BDT_PER_USD);
const form = document.querySelector('#homeForm');
const button = document.querySelector('#submitBtn');

async function loadSummary(){
  try {
    const response = await fetch('/api/summary');
    const data = await response.json();
    if (!data.ready) throw new Error(data.error || 'Model is not ready');
    for (const [field, options] of Object.entries(data.category_options)) {
      const select = form.elements.namedItem(field);
      select.replaceChildren(...options.map(value => new Option(value.replace(/\b\w/g, char => char.toUpperCase()), value)));
    }
    const notice = document.querySelector('#modelNotice');
    notice.hidden = false;
    notice.querySelector('div').innerHTML = `<strong>Your housing data is loaded</strong><br>${data.rows.toLocaleString()} cleaned sale records used for training and evaluation.`;
    document.querySelector('#accuracy').textContent = `${(data.metrics.accuracy*100).toFixed(1)}%`;
    document.querySelector('#f1').textContent = data.metrics.macro_f1.toFixed(2);
    document.querySelector('#testRows').textContent = data.metrics.test_rows.toLocaleString();
  } catch (error) {
    document.querySelector('#modelNotice').hidden = false;
    document.querySelector('#modelNotice').innerHTML = `<span>!</span><div><strong>Model unavailable</strong><br>${error.message}</div>`;
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.querySelector('span:first-child').textContent = 'Estimating…';
  document.querySelector('#formError').hidden = true;
  const values = Object.fromEntries(new FormData(form));
  for (const key of ['area','bedrooms','bathrooms','stories','parking']) values[key] = Number(values[key]);
  try {
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not create estimate');
    document.querySelector('#formPrice').textContent = formatPrice(data.price_estimate);
    document.querySelector('#formBand').textContent = `${data.price_band} price band · ${Math.round(data.confidence*100)}% confidence`;
    document.querySelector('#formResult').hidden = false;
    document.querySelector('#emptyState').hidden = true;
    document.querySelector('#resultState').hidden = false;
    document.querySelector('#price').textContent = formatPrice(data.price_estimate);
    document.querySelector('#band').textContent = `${data.price_band} home`;
    document.querySelector('#confidence').textContent = `${Math.round(data.confidence*100)}% top-band confidence`;
    const order = ['Budget','Standard','Premium','Luxury'];
    document.querySelector('#probabilities').innerHTML = order.map(label => {
      const value = data.band_probabilities[label] ?? 0;
      return `<div class="prob-row"><span>${label}</span><div class="track"><div class="fill" style="width:${value*100}%"></div></div><span class="prob-value">${Math.round(value*100)}%</span></div>`;
    }).join('');
    document.querySelector('#formResult').scrollIntoView({behavior:'smooth', block:'nearest'});
  } catch (error) {
    const message = document.querySelector('#formError');
    message.textContent = `Could not estimate price: ${error.message}`;
    message.hidden = false;
  } finally {
    button.disabled = false;
    button.querySelector('span:first-child').textContent = 'Estimate home value';
  }
});

loadSummary();
