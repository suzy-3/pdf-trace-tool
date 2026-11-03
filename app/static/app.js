const $ = id => document.getElementById(id);
let token = null, busy = false;
const base = () => `/api/documents/${token}/pages/${$('page').value}`;
function status(text) { $('status').textContent = text; }
function setBusy(value) {
  busy = value;
  for (const id of ['file', 'page', 'process', 'close']) $(id).disabled = value || (id !== 'file' && !token);
}
function resetOutput() {
  $('corrected').hidden = true;
  $('empty-corrected').hidden = false;
  for (const id of ['pdf', 'svg']) {
    $(id).removeAttribute('href'); $(id).classList.add('disabled'); $(id).setAttribute('aria-disabled', 'true');
  }
}
async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : '処理に失敗しました。入力内容を確認してください。');
  }
  return response;
}
function showPage() {
  resetOutput(); $('original').hidden = false; $('empty-original').hidden = true;
  $('original').src = `${base()}/original.png`;
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
  setBusy(true); resetOutput(); status('図面を補正して線を抽出しています…');
  try {
    const data = await (await request(`${base()}/process`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({deskew: $('deskew').checked, denoise: $('denoise').checked, sharpen: $('sharpen').checked, min_area: Number($('area').value)})})).json();
    $('corrected').src = `${base()}/corrected.png?t=${Date.now()}`;
    $('corrected').hidden = false; $('empty-corrected').hidden = true;
    for (const [id, name] of [['pdf', 'corrected.pdf'], ['svg', 'trace.svg']]) {
      $(id).href = `${base()}/${name}`; $(id).download = `page-${$('page').value}.${id}`;
      $(id).classList.remove('disabled'); $(id).setAttribute('aria-disabled', 'false');
    }
    status(`変換しました。検出した傾き: ${data.angle}° ／ ${data.width} × ${data.height} px`);
  } catch (error) { status(error.message); }
  finally { setBusy(false); }
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
