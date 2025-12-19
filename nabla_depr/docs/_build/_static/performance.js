// Performance and Core Web Vitals optimization
(function() {
    'use strict';
    
    // Preload critical resources
    function preloadCriticalResources() {
        const criticalResources = [
            { href: '/_static/custom.css', as: 'style' },
            { href: '/_static/nabla-logo.png', as: 'image' }
        ];
        
        criticalResources.forEach(resource => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.href = resource.href;
            link.as = resource.as;
            if (resource.as === 'image') {
                link.type = 'image/png';
            }
            document.head.appendChild(link);
        });
    }
    
    // Optimize image loading
    function optimizeImages() {
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            // Add loading="lazy" if not already present
            if (!img.hasAttribute('loading')) {
                img.setAttribute('loading', 'lazy');
            }
            
            // Add decoding="async" for better performance
            if (!img.hasAttribute('decoding')) {
                img.setAttribute('decoding', 'async');
            }
            
            // Add proper alt text if missing (for accessibility and SEO)
            if (!img.hasAttribute('alt') || !img.alt.trim()) {
                const figcaption = img.closest('figure')?.querySelector('figcaption');
                if (figcaption) {
                    img.alt = figcaption.textContent.trim();
                } else {
                    img.alt = 'Nabla documentation image';
                }
            }
        });
    }
    
    // Service Worker for caching (optional, improves return visits)
    function registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            // Only register in production
            if (location.hostname === 'nablaml.com') {
                navigator.serviceWorker.register('/sw.js')
                    .then(registration => {
                        console.log('SW registered:', registration);
                    })
                    .catch(error => {
                        console.log('SW registration failed:', error);
                    });
            }
        }
    }
    
    // Optimize third-party scripts
    function optimizeThirdPartyScripts() {
        // Defer non-critical scripts
        const scripts = document.querySelectorAll('script[src]');
        scripts.forEach(script => {
            if (!script.hasAttribute('async') && !script.hasAttribute('defer')) {
                // Defer scripts that aren't explicitly marked as async
                if (!script.src.includes('gtag') && !script.src.includes('analytics')) {
                    script.defer = true;
                }
            }
        });
    }
    
    // Web Vitals monitoring and optimization
    function optimizeWebVitals() {
        // Reduce layout shift by setting dimensions on images
        const images = document.querySelectorAll('img:not([width]), img:not([height])');
        images.forEach(img => {
            if (img.naturalWidth && img.naturalHeight) {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;
                img.style.width = 'auto';
                img.style.height = 'auto';
                img.style.maxWidth = '100%';
            }
        });
        
        // Optimize LCP by preloading hero images
        const heroImage = document.querySelector('.hero img, h1 + * img, .bd-content img:first-of-type');
        if (heroImage && heroImage.src) {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'image';
            link.href = heroImage.src;
            document.head.appendChild(link);
        }
    }
    
    // Resource hints for external domains
    function addResourceHints() {
        const hints = [
            { rel: 'dns-prefetch', href: '//fonts.googleapis.com' },
            { rel: 'dns-prefetch', href: '//github.com' },
            { rel: 'dns-prefetch', href: '//pypi.org' },
            { rel: 'preconnect', href: 'https://fonts.googleapis.com', crossorigin: '' },
            { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }
        ];
        
        hints.forEach(hint => {
            const link = document.createElement('link');
            link.rel = hint.rel;
            link.href = hint.href;
            if (hint.crossorigin !== undefined) {
                link.crossOrigin = hint.crossorigin;
            }
            document.head.appendChild(link);
        });
    }
    
    // Initialize optimizations
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            preloadCriticalResources();
            optimizeImages();
            optimizeThirdPartyScripts();
            optimizeWebVitals();
            addResourceHints();
            registerServiceWorker();
        });
    } else {
        preloadCriticalResources();
        optimizeImages();
        optimizeThirdPartyScripts();
        optimizeWebVitals();
        addResourceHints();
        registerServiceWorker();
    }
    
    // Performance monitoring
    window.addEventListener('load', function() {
        // Log performance metrics for debugging
        if (window.performance && window.performance.timing) {
            const timing = window.performance.timing;
            const loadTime = timing.loadEventEnd - timing.navigationStart;
            const domReady = timing.domContentLoadedEventEnd - timing.navigationStart;
            
            console.log('Page Performance:', {
                loadTime: loadTime + 'ms',
                domReady: domReady + 'ms',
                firstPaint: window.performance.getEntriesByType('paint')[0]?.startTime + 'ms'
            });
        }
    });
})();
