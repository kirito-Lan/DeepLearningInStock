<template>
  <div id="basicLayout">
    <a-layout style="min-height: 100vh">
      <a-layout-header class="header">
        <GlobalHeader />
      </a-layout-header>

      <a-layout-content class="content">
        <router-view />
      </a-layout-content>

      <a-layout-footer class="footer">
        <a href="https://data.stats.gov.cn/index.htm" target="_blank"> Stock Reasoning @Kirito </a>
      </a-layout-footer>
    </a-layout>
  </div>
</template>

<script setup lang="ts">
import GlobalHeader from '@/components/GlobalHeader.vue'
</script>

<style scoped>
#basicLayout .header {
  left: 0;
  right: 0;
  position: fixed;
  padding-inline: 0;
  margin-bottom: 16px;
  color: unset;
  background: #ffffff;
  width: 100%;
  z-index: 1000;
}

#basicLayout .content {
  left: 0;
  background: linear-gradient(to right, #fefefe, #fff);
  margin-bottom: 28px;
  padding: 20px;
  margin-top: 64px;
  z-index: 1;
}

#basicLayout .footer {
  background: #efefef;
  padding: 16px;
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  text-align: center;
  z-index: 1;
}
</style>
