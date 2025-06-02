// Collapsible TOC (Table of Contents) functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create the toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'toc-toggle-btn';
    toggleBtn.innerHTML = 'TOC';
    toggleBtn.setAttribute('aria-label', 'Toggle Table of Contents');
    toggleBtn.setAttribute('title', 'Toggle Table of Contents');
    
    // Find the TOC sidebar
    const tocSidebar = document.querySelector('.bd-toc');
    const mainContent = document.querySelector('.bd-main');
    
    if (tocSidebar && mainContent) {
        // Add the toggle button to the page
        document.body.appendChild(toggleBtn);
        
        // Check if user preference exists in localStorage
        const tocHidden = localStorage.getItem('nabla-toc-hidden') === 'true';
        
        // Apply initial state
        if (tocHidden) {
            tocSidebar.classList.add('toc-hidden');
            mainContent.classList.add('toc-hidden');
            toggleBtn.innerHTML = 'Show TOC';
        }
        
        // Toggle functionality
        toggleBtn.addEventListener('click', function() {
            const isHidden = tocSidebar.classList.contains('toc-hidden');
            
            if (isHidden) {
                // Show TOC
                tocSidebar.classList.remove('toc-hidden');
                mainContent.classList.remove('toc-hidden');
                toggleBtn.innerHTML = 'TOC';
                localStorage.setItem('nabla-toc-hidden', 'false');
            } else {
                // Hide TOC
                tocSidebar.classList.add('toc-hidden');
                mainContent.classList.add('toc-hidden');
                toggleBtn.innerHTML = 'Show TOC';
                localStorage.setItem('nabla-toc-hidden', 'true');
            }
        });
        
        // Keyboard accessibility
        toggleBtn.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleBtn.click();
            }
        });
    }
    
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
