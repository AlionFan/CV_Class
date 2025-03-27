document.addEventListener('DOMContentLoaded', () => {
    // 为所有代码块添加复制按钮
    document.querySelectorAll('pre').forEach(pre => {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = '复制';
        pre.appendChild(btn);
        
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                btn.textContent = '已复制!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = '复制';
                    btn.classList.remove('copied');
                }, 2000);
            });
        });
    });
});