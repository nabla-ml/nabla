window.addEventListener("DOMContentLoaded", function () {
    const themeSwitcher = document.querySelector(".theme-switch-button");
    if (themeSwitcher) {
        themeSwitcher.remove();
    }

    const pageName = window.DOCUMENTATION_OPTIONS?.pagename;
    if (pageName === "index") {
        document.body.classList.add("home-landing");
    }
});
