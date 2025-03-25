---
layout: default
title: 实验课
---

<script src="/CV_Class/assets/js/page_nav.js"></script>

<script src="/CV_Class/assets/js/content_nav.js"></script>

<div class="dataset-page">
    <!-- 左侧导航栏 - 页面间跳转 -->
    <div class="dataset-sidebar">
        <ul class="dataset-nav">
            <li><a href="#lab1" class="active" data-content="work1">1.第一次课</a></li>
            <li><a href="#lab2" data-content="work2">2. 第二次课</a></li>
        </ul>
    </div>

    <!-- 中间内容区域 -->
    <div class="dataset-content">
        <div id="work1-content" class="work-content active">
            {% include_relative labs/lab1/lab1.html %}
        </div>

        <div id="work2-content" class="work-content">
            {% include_relative labs/lab2/lab2.html %}
        </div>

    </div>
    
    <!-- 右侧目录导航 - 页面内跳转 -->
    <div class="content-sidebar">
        <div class="toc-container">
            <h3>目录</h3>
            <div id="toc" class="toc-nav"></div>
        </div>
    </div>
</div>




