(function(){
  const form = document.getElementById('reviewForm');
  if(!form) return;

  const QUEUE_KEY = 'anpr_offline_actions';

  function loadQueue(){
    try{return JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]')}catch{return []}
  }
  function saveQueue(q){localStorage.setItem(QUEUE_KEY, JSON.stringify(q));}

  async function flushQueue(){
    if(!navigator.onLine) return;
    const q = loadQueue();
    if(!q.length) return;
    const resp = await fetch('/api/sync-actions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({actions:q})});
    if(resp.ok){saveQueue([]);}
  }

  window.addEventListener('online', flushQueue);
  setTimeout(flushQueue, 800);

  form.addEventListener('submit', async (e)=>{
    if(navigator.onLine) return;
    e.preventDefault();

    const fd = new FormData(form);
    const data = {
      event_id: parseInt(location.pathname.split('/')[2], 10),
      action: fd.get('action'),
      corrected_plate: fd.get('corrected_plate') || '',
      best_image: fd.get('best_image') || '',
      source: fd.get('source') || 'dashboard',
      admin_password: fd.get('admin_password') || '',
      sub_parent_id: fd.get('sub_parent_id') || '',
      sub_plate: fd.get('sub_plate') || ''
    };

    const q = loadQueue();
    q.push(data);
    saveQueue(q);
    alert('No internet. Action saved offline and will sync automatically.');
    location.href = '/dashboard';
  });
})();
