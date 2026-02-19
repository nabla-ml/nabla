window.addEventListener("DOMContentLoaded", function () {
    // Remove theme switcher
    const themeSwitcher = document.querySelector(".theme-switch-button");
    if (themeSwitcher) themeSwitcher.remove();

    const pageName = window.DOCUMENTATION_OPTIONS?.pagename;
    const isIndex = pageName === "index";

    if (isIndex) {
        document.body.classList.add("home-landing");
    }

    const topHeader = document.querySelector("header.bd-header");
    if (topHeader && !topHeader.querySelector(".nabla-topbar")) {

        // Capture the article-header buttons ONLY (they live in bd-header-article, not the sidebar)
        const articleButtons = document.querySelector(".article-header-buttons");

        // Compute relative path back to root reliably from current page depth
        // e.g. /api/index.html → depth=1 → "../"
        // e.g. /examples/01_foo.html → depth=1 → "../"
        // e.g. /index.html → depth=0 → "./"
        const pathParts = window.location.pathname.replace(/\/+$/, '').split('/').filter(Boolean);
        const depth = pathParts.length > 0 ? pathParts.length - 1 : 0;
        const rootPrefix = depth > 0 ? '../'.repeat(depth) : './';
        const homeHref = rootPrefix + 'index.html';
        const searchHref = rootPrefix + 'search.html';

        topHeader.innerHTML = `
            <div class="nabla-topbar">
                <div class="nabla-topbar-left">
                    ${!isIndex ? `<button class="nabla-sidebar-toggle btn btn-sm" id="nabla-sidebar-toggle-btn" title="Toggle sidebar" aria-label="Toggle sidebar">
                        <span class="fa-solid fa-bars"></span>
                    </button>` : ""}
                    <a href="${homeHref}" class="nabla-topbar-brand">Nabla</a>
                </div>
                <div class="nabla-topbar-actions">
                    <a href="${searchHref}" class="nabla-search-link btn btn-sm" title="Search" aria-label="Search">
                        <i class="fa-solid fa-magnifying-glass"></i>
                    </a>
                </div>
            </div>
        `;

        // Wire up our custom sidebar toggle to the Sphinx checkbox
        const toggleBtn = topHeader.querySelector("#nabla-sidebar-toggle-btn");
        const sidebarCheckbox = document.getElementById("pst-primary-sidebar-checkbox");
        if (toggleBtn && sidebarCheckbox) {
            toggleBtn.addEventListener("click", function () {
                sidebarCheckbox.checked = !sidebarCheckbox.checked;
            });
        }

        // Move the article-level source/edit/issue buttons into the topbar (they are NOT in the sidebar)
        // Strip anything we don't want before moving
        const topbarActions = topHeader.querySelector(".nabla-topbar-actions");
        if (articleButtons && topbarActions) {
            // Remove search button duplicate (we already have nabla-search-link)
            articleButtons.querySelectorAll(".search-button, .pst-navbar-icon").forEach(el => el.remove());
            // Remove secondary sidebar toggle
            articleButtons.querySelectorAll(".secondary-toggle").forEach(el => el.remove());
            // Remove theme switch
            articleButtons.querySelectorAll(".theme-switch-button").forEach(el => el.remove());

            // Replace the GitHub source dropdown with a single direct link to the repo
            const sourceDropdown = articleButtons.querySelector(".dropdown-source-buttons");
            if (sourceDropdown) {
                const repoLink = sourceDropdown.querySelector(".btn-source-repository-button");
                const repoHref = repoLink ? repoLink.href : "https://github.com/nabla-ml/nabla";
                const ghBtn = document.createElement("a");
                ghBtn.href = repoHref;
                ghBtn.target = "_blank";
                ghBtn.rel = "noopener noreferrer";
                ghBtn.className = "nabla-gh-link btn btn-sm";
                ghBtn.title = "GitHub repository";
                ghBtn.setAttribute("aria-label", "GitHub repository");
                ghBtn.innerHTML = '<i class="fab fa-github"></i>';
                sourceDropdown.replaceWith(ghBtn);
            }

            topbarActions.appendChild(articleButtons);
        }
    }

    // Hide the now-empty secondary article header bar
    const articleHeader = document.querySelector(".bd-header-article");
    if (articleHeader) articleHeader.style.display = "none";
});
