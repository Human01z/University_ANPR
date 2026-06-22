 (function(){
  const key = 'anpr_unread_alert_enabled';
  const toggle = document.getElementById('unreadAlertToggle');
  const alertBox = document.getElementById('unreadAlert');

  function enabled(){
    return localStorage.getItem(key) !== 'false';
  }

  function render(){
    if(toggle) toggle.checked = enabled();
    if(alertBox){
      const count = parseInt(alertBox.dataset.count || '0', 10);
      alertBox.style.display = enabled() && count > 0 ? 'block' : 'none';
    }
  }

  if(toggle){
    toggle.addEventListener('change', function(){
      localStorage.setItem(key, toggle.checked ? 'true' : 'false');
      render();
    });
  }

  render();
})();
