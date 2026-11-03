const $ = id => document.getElementById(id);
let token = null, busy = false;
const base = () => `/api/documents/${token}/pages/${$('page').value}`;
function status(text) { $('status').textContent = text; }
function setBusy(value) {
  busy = value;
  for (const id of ['mode', 'noise-strength', 'deskew', 'denoise', 'sharpen', 'area']) $(id).disabled = value;
  for (const id of ['file', 'page', 'process', 'close']) $(id).disabled = value || (id !== 'file' && !token);
}
function resetOutput() {
  $('review').hidden = true;
  $('preserved').removeAttribute('href'); $('preserved').classList.add('disabled'); $('preserved').setAttribute('aria-disabled', 'true');
  $('corrected').hidden = true;
  $('empty-corrected').hidden = false;
  for (const id of ['pdf', 'svg']) {
    $(id).removeAttribute('href'); $(id).classList.add('disabled'); $(id).setAttribute('aria-disabled', 'true');
  }
}
async function request(url, options) {
  let response;
  try { response = await fetch(url, options); }
  catch { throw new Error('通信が切れました。処理が継続している場合があります。再接続後に処理状況を確認してください。'); }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : '処理に失敗しました。入力内容を確認してください。');
  }
  return response;
}
function showPage() {
  resetOutput(); $('original').hidden = false; $('empty-original').hidden = true;
  $('original').src = `${base()}/original.png`;
  enablePreserved();
  status(`${$('page').value}ページ目を表示しています。補正・SVG変換を実行してください。`);
}
$('original').onerror = () => status('ページを表示できませんでした。PDFを再アップロードしてください。');
$('file').onchange = async () => {
  const file = $('file').files[0]; if (!file || busy) return;
  setBusy(true); status('PDFを読み込んでいます…');
  try {
    if (token) { await request(`/api/documents/${token}`, {method: 'DELETE'}); token = null; }
    resetOutput(); $('original').hidden = true; $('empty-original').hidden = false;
    const form = new FormData(); form.append('file', file);
    const data = await (await request('/api/documents', {method: 'POST', body: form})).json();
    token = data.id; $('page').replaceChildren();
    for (let i = 1; i <= data.pages; i++) $('page').add(new Option(`${i} / ${data.pages}`, i));
    showPage();
  } catch (error) { status(error.message); }
  finally { $('file').value = ''; setBusy(false); }
};
$('page').onchange = showPage;
$('process').onclick = async () => {
  setBusy(true); resetOutput(); enablePreserved(); status('図面を補正して線を抽出しています…');
  try {
    const jobBase = base();
    let job = await (await request(`${jobBase}/jobs`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode: $('mode').value, noise_strength: Number($('noise-strength').value), deskew: $('deskew').checked, denoise: $('denoise').checked, sharpen: $('sharpen').checked, min_area: Number($('area').value)})})).json();
    $('progress').hidden = false;
    while (job.state === 'running') {
      $('progress').value = job.percent;
      status(`${job.stage}: ${job.percent}% ／ 小領域 ${job.completed} / ${job.total}`);
      await new Promise(resolve => setTimeout(resolve, 350));
      job = await (await request(`${jobBase}/jobs/${job.id}`)).json();
    }
    if (job.state === 'failed') throw new Error(job.error);
    const data = job.result;
    $('corrected').src = `${base()}/corrected.png?t=${Date.now()}`;
    $('corrected').hidden = false; $('empty-corrected').hidden = true;
    for (const [id, name] of [['pdf', 'corrected.pdf'], ['svg', 'trace.svg']]) {
      $(id).href = `${base()}/${name}`; $(id).download = `page-${$('page').value}.${id}`;
      $(id).classList.remove('disabled'); $(id).setAttribute('aria-disabled', 'false');
    }
    if (data.mode === 'cad_clean') {
      const stamp = Date.now();
      $('reference').src = `${base()}/reference.png?t=${stamp}`;
      $('overlay').src = `${base()}/corrected.png?t=${stamp}`;
      $('uncertain').src = `${base()}/review.png?t=${stamp}`;
      $('difference').src = `${base()}/difference.png?t=${stamp}`;
      $('review').hidden = false; updateReview();
    }
    status(`変換しました。${data.warning || ''} ${data.seconds !== undefined ? `${data.seconds}秒 / 原画保持 ${Math.round((data.fallback_ratio || 0) * 100)}% / 欠落候補 ${data.missing_pixels}px / 追加候補 ${data.added_pixels}px。` : ''}検出した傾き: ${data.angle}° ／ ${data.width} × ${data.height} px`);
  } catch (error) { status(error.message); }
  finally { $('progress').hidden = true; setBusy(false); }
};
$('close').onclick = async () => {
  setBusy(true);
  try {
    await request(`/api/documents/${token}`, {method: 'DELETE'}); token = null;
    $('page').replaceChildren(); $('original').hidden = true; $('empty-original').hidden = false;
    resetOutput(); status('図面を削除しました。次のPDFを選択してください。');
  } catch (error) { status(error.message); }
  finally { setBusy(false); }
};

function updateReview() {
  const diff = $('review-mode').value === 'difference';
  const uncertain = $('review-mode').value === 'uncertain';
  $('uncertain').hidden = !uncertain;
  $('difference').hidden = !diff;
  $('reference').hidden = diff || uncertain; $('overlay').hidden = diff || uncertain;
  $('overlay').style.opacity = Number($('opacity').value) / 100;
}
$('opacity').oninput = updateReview;
$('review-mode').onchange = updateReview;
for (const id of ['mode', 'noise-strength', 'deskew', 'denoise', 'sharpen', 'area']) {
  $(id).onchange = () => { resetOutput(); enablePreserved(); syncControls(); if (token) status('設定を変更しました。再変換してください。'); };
}

function enablePreserved() {
  if (!token) return;
  $('preserved').href = `${base()}/preserved.pdf`; $('preserved').download = `page-${$('page').value}-preserved.pdf`;
  $('preserved').classList.remove('disabled'); $('preserved').setAttribute('aria-disabled', 'false');
}

function syncControls() {
  const cad = $('mode').value === 'cad_clean';
  for (const id of ['denoise', 'sharpen', 'area']) $(id).closest('label').hidden = cad;
  $('noise-strength').closest('label').hidden = !cad;
}
syncControls();
