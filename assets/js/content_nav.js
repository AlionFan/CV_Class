document.addEventListener('DOMContentLoaded', function() {
    // 原有的标签页切换功能
    const navLinks = document.querySelectorAll('.dataset-nav a');
    const contentDivs = document.querySelectorAll('.work-content');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            navLinks.forEach(a => a.classList.remove('active'));
            this.classList.add('active');

            contentDivs.forEach(div => div.classList.remove('active'));
            
            const contentId = this.getAttribute('data-content') + '-content';
            document.getElementById(contentId).classList.add('active');
            
            // 切换标签页后重新生成目录
            generateTOC();
        });
    });
    
    // 生成目录功能
    function generateTOC() {
        const activeContent = document.querySelector('.work-content.active');
        if (!activeContent) return;
        
        const toc = document.getElementById('toc');
        toc.innerHTML = '';
        
        const headings = activeContent.querySelectorAll('h1, h2, h3, h4');
        const tocList = document.createElement('ul');
        
        headings.forEach((heading, index) => {
            // 为每个标题创建ID（如果没有）
            if (!heading.id) {
                heading.id = 'heading-' + index;
            }
            
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            
            link.href = '#' + heading.id;
            link.textContent = heading.textContent;
            link.classList.add('toc-' + heading.tagName.toLowerCase());
            
            link.addEventListener('click', function(e) {
                e.preventDefault();
                heading.scrollIntoView({behavior: 'smooth'});
            });
            
            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });
        
        toc.appendChild(tocList);
    }
    
    // 初始生成目录
    generateTOC();
});