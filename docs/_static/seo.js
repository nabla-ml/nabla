// SEO enhancements for Nabla documentation
// Adds structured data and performance optimizations

document.addEventListener('DOMContentLoaded', function() {
    // Add structured data for search engines
    const structuredData = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "Nabla",
        "description": "Python library for GPU-accelerated array computation with NumPy-like API, JAX-style transformations (vmap, grad, jit), and Mojo integration",
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Cross-platform",
        "programmingLanguage": ["Python", "Mojo"],
        "url": "https://nablaml.com",
        "downloadUrl": "https://pypi.org/project/nabla-ml/",
        "author": {
            "@type": "Organization",
            "name": "Nabla Team"
        },
        "license": "MIT",
        "keywords": ["python", "arrays", "gpu", "numpy", "jax", "mojo", "machine learning", "automatic differentiation"]
    };

    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.textContent = JSON.stringify(structuredData);
    document.head.appendChild(script);

    // Add canonical URL if not present
    if (!document.querySelector('link[rel="canonical"]')) {
        const canonical = document.createElement('link');
        canonical.rel = 'canonical';
        canonical.href = window.location.href;
        document.head.appendChild(canonical);
    }

    // Optimize images for SEO (add alt text if missing)
    document.querySelectorAll('img:not([alt])').forEach(img => {
        img.alt = 'Nabla documentation image';
    });
});
