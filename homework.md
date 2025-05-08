---
layout: default
title: 每周作业
---

<script src="/CV_Class/assets/js/page_nav.js"></script>

<script src="/CV_Class/assets/js/content_nav.js"></script>

<script src="/CV_Class/assets/js/mobile-nav.js"></script>

<div class="dataset-page">
    <!-- 左侧导航栏 - 页面间跳转 -->
    <div class="dataset-sidebar">
        <ul class="dataset-nav">
            <li><a href="#work1" class="active" data-content="work1">1. First Attempt</a></li>
            <li><a href="#work2" data-content="work2">2. Edge Detection</a></li>
            <li><a href="#work3" data-content="work3">3. Fingerprint matching and image stitching</a></li>
            <li><a href="#work4" data-content="work4">4. Target Tracking</a></li>
        </ul>
    </div>

    <!-- 中间内容区域 -->
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

        <div id="work4-content" class="work-content">
            {% include_relative works/work4/work4.html %}
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




