// script.js
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('overlay');
    const overlayImg = document.getElementById('overlay-img');

    document.querySelectorAll('.chart img').forEach(img => {
        img.addEventListener('click', () => {
            overlayImg.src = img.src;
            overlay.style.display = 'flex';
        });
    });

    overlay.addEventListener('click', () => {
        overlay.style.display = 'none';
    });
});
