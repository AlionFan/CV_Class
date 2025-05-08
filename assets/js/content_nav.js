document.addEventListener('DOMContentLoaded', function() {
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
    
    // 当标签页切换时重新生成目录
    const navLinks = document.querySelectorAll('.dataset-nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            setTimeout(generateTOC, 100); // 延迟一点时间确保内容已切换
        });
    });
});