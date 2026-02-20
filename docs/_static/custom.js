(() => {
    const earlyPageName = window.DOCUMENTATION_OPTIONS?.pagename;
    const earlyIsIndex = earlyPageName === "index" || window.location.pathname.endsWith("/index.html") || window.location.pathname === "/";

    if (earlyIsIndex) {
        document.documentElement.classList.add("home-landing");
    }

    function initNablaUi() {
        const pageName = window.DOCUMENTATION_OPTIONS?.pagename;
        const isIndex = pageName === "index" || earlyIsIndex;

        if (isIndex && document.body) {
            document.body.classList.add("home-landing");
        }

        const topHeader = document.querySelector("header.bd-header");
        if (topHeader && !topHeader.querySelector(".nabla-topbar")) {

            const articleButtons = document.querySelector(".article-header-buttons");

            const pathParts = window.location.pathname.replace(/\/+$/, "").split("/").filter(Boolean);
            const depth = pathParts.length > 0 ? pathParts.length - 1 : 0;
            const rootPrefix = depth > 0 ? "../".repeat(depth) : "./";
            const homeHref = rootPrefix + "index.html";
            const searchHref = rootPrefix + "search.html";

            topHeader.innerHTML = `
                <div class="nabla-topbar">
                    <div class="nabla-topbar-left">
                        <button class="nabla-sidebar-toggle btn btn-sm" id="nabla-sidebar-toggle-btn" title="Toggle sidebar" aria-label="Toggle sidebar">
                            <span class="fa-solid fa-bars"></span>
                        </button>
                        <a href="${homeHref}" class="nabla-topbar-brand">Nabla</a>
                    </div>
                    <div class="nabla-topbar-actions">
                        <a href="${searchHref}" class="nabla-search-link btn btn-sm" title="Search" aria-label="Search">
                            <i class="fa-solid fa-magnifying-glass"></i>
                        </a>
                    </div>
                </div>
            `;

            const toggleBtn = topHeader.querySelector("#nabla-sidebar-toggle-btn");
            const sidebarCheckbox = document.getElementById("pst-primary-sidebar-checkbox");
            if (toggleBtn && sidebarCheckbox) {
                toggleBtn.addEventListener("click", function () {
                    sidebarCheckbox.checked = !sidebarCheckbox.checked;
                });
            }

            const topbarActions = topHeader.querySelector(".nabla-topbar-actions");
            if (articleButtons && topbarActions) {
                articleButtons.querySelectorAll(".search-button, .pst-navbar-icon").forEach((el) => el.remove());
                articleButtons.querySelectorAll(".secondary-toggle").forEach((el) => el.remove());
                articleButtons.querySelectorAll(".theme-switch-button").forEach((el) => el.remove());

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

        const articleHeader = document.querySelector(".bd-header-article");
        if (articleHeader) articleHeader.style.display = "none";
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initNablaUi, { once: true });
    } else {
        initNablaUi();
    }
})();
