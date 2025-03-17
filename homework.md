---
layout: default
title: Dataset
---

<div class="dataset-page">
    <div class="dataset-sidebar">
        <ul class="dataset-nav">
            <li><a href="#work1" class="active" data-content="work1">1. First Attempt</a></li>
            <li><a href="#work2" data-content="work2">2. Edge Detection</a></li>
            <li><a href="#work3" data-content="work3">3. Future Work</a></li>
        </ul>
    </div>

    <div class="dataset-content">
        <div id="work1-content" class="work-content active">
            {% include_relative works/work1/work1.html %}
        </div>

        <div id="work2-content" class="work-content">
            {% include_relative works/work2/work2.html %}
        </div>

        <div id="work3-content" class="work-content">
            {% include_relative works/work3/work3.html %}
        </div>
    </div>
</div>

<script>
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
</script>

<footer style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f5f5f5;">
    <p>© Copyright Capital Normal University 2025</p>
</footer>


