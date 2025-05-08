document.addEventListener('DOMContentLoaded', function() {
    // 创建汉堡菜单按钮
    const menuToggle = document.createElement('button');
    menuToggle.className = 'menu-toggle';
    menuToggle.innerHTML = '<span></span><span></span><span></span>';
    document.body.appendChild(menuToggle);

    // 获取侧边栏
    const sidebar = document.querySelector('.dataset-sidebar');
    const contentSidebar = document.querySelector('.content-sidebar');

    // 点击汉堡菜单按钮切换侧边栏
    menuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
        // 当侧边栏打开时，添加遮罩层
        if (sidebar.classList.contains('active')) {
            createOverlay();
        } else {
            removeOverlay();
        }
    });

    // 创建遮罩层
    function createOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
        overlay.addEventListener('click', function() {
            sidebar.classList.remove('active');
            removeOverlay();
        });
    }

    // 移除遮罩层
    function removeOverlay() {
        const overlay = document.querySelector('.sidebar-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    // 处理窗口大小变化
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            sidebar.classList.remove('active');
            removeOverlay();
        }
    });

    // 优化移动端滚动
    if ('ontouchstart' in window) {
        document.body.style.overflow = 'hidden';
        document.body.style.position = 'fixed';
        document.body.style.width = '100%';
        document.body.style.top = `-${window.scrollY}px`;
    }
}); 