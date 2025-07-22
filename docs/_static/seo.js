// Essential SEO enhancements only
document.addEventListener('DOMContentLoaded', function() {
    // Add canonical URL if missing (critical for SEO)
    if (!document.querySelector('link[rel="canonical"]')) {
        const canonical = document.createElement('link');
        canonical.rel = 'canonical';
        canonical.href = window.location.href;
        document.head.appendChild(canonical);
    }

    // Add missing alt text to images (accessibility + SEO)
    document.querySelectorAll('img:not([alt])').forEach(img => {
        img.alt = 'Nabla documentation';
    });

    // Add heading IDs for anchor linking (improves internal linking)
    document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(heading => {
        if (!heading.id && heading.textContent) {
            heading.id = heading.textContent.toLowerCase()
                .replace(/[^\w\s-]/g, '')
                .replace(/\s+/g, '-')
                .trim();
        }
    });
});
