document.addEventListener('DOMContentLoaded', function() {
    // 获取所有导航链接和内容区域
    const navLinks = document.querySelectorAll('.dataset-nav a');
    const contentDivs = document.querySelectorAll('.work-content');

    // 为每个导航链接添加点击事件
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // 移除所有导航链接的active类
            navLinks.forEach(a => a.classList.remove('active'));
            // 为当前点击的链接添加active类
            this.classList.add('active');

            // 隐藏所有内容
            contentDivs.forEach(div => div.classList.remove('active'));
            
            // 显示对应的内容
            const contentId = this.getAttribute('data-content') + '-content';
            document.getElementById(contentId).classList.add('active');
        });
    });
});