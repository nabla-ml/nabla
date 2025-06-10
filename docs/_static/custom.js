/* =================================================================
   NABLA DOCUMENTATION - CUSTOM JAVASCRIPT
   Force dark mode only and enhanced search
   ================================================================= */

document.addEventListener('DOMContentLoaded', function() {
    // Force dark mode immediately
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.setAttribute('data-theme', 'dark');
    
    // Hide any theme switcher buttons
    const themeSwitchers = document.querySelectorAll('.theme-switch, .theme-toggle, .bd-theme-toggle, button[data-mode-switch]');
    themeSwitchers.forEach(switcher => {
        switcher.style.display = 'none';
    });
    
    // Prevent theme switching
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                if (mutation.target.getAttribute('data-theme') !== 'dark') {
                    mutation.target.setAttribute('data-theme', 'dark');
                }
            }
        });
    });
    
    observer.observe(document.documentElement, { attributes: true });
    observer.observe(document.body, { attributes: true });

    // Enhanced search functionality
    const searchInput = document.querySelector('.bd-search input[type="search"]');
    if (searchInput) {
        // Add placeholder text
        searchInput.placeholder = 'Search documentation...';
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
            
            // Escape to blur search
            if (e.key === 'Escape' && document.activeElement === searchInput) {
                searchInput.blur();
            }
        });
    }
});
