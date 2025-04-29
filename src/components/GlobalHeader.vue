<template>
  <div id="GlobalHeader">
    <a-row :wrap="false">
      <a-col flex="300px">
        <RouterLink to="/">
          <div class="title-bar">
            <img class="logo" src="../assets/favicon.ico" alt="logo" />
            <div class="title">Stock&Macro Reasoning</div>
          </div>
        </RouterLink>
      </a-col>
      <a-col flex="auto">
        <a-menu
          v-model:selectedKeys="current"
          mode="horizontal"
          :items="items"
          @click="doMenuClick"
        />
      </a-col>
    </a-row>
  </div>
</template>
<script lang="ts" setup>
import { h, ref } from 'vue'
import {
  BarChartOutlined,
  StockOutlined,
  BgColorsOutlined,
  GithubOutlined,
} from '@ant-design/icons-vue'
import { MenuProps } from 'ant-design-vue'
import { useRouter } from 'vue-router'
const router = useRouter()

const current = ref<string[]>([])
// 监听路由变化，更新当前选中菜单
router.afterEach((to, from, next) => {
  current.value = [to.path]
})

// 路由跳转事件
const doMenuClick = ({ key }: { key: string }) => {
  router.push({
    path: key,
  })
}

const items = ref<MenuProps['items']>([
  {
    key: '/',
    icon: () => h(BarChartOutlined),
    label: '宏观数据',
    title: '宏观数据看板',
  },
  {
    key: '/exponent',
    icon: () => h(StockOutlined),
    label: '指标数据',
    title: '指标数据看板',
  },
  {
    key: '/predict',
    icon: () => h(BgColorsOutlined),
    label: '数据预测',
    title: '数据预测看板',
  },
  {
    key: '/about',
    icon: () => h(GithubOutlined),
    label: h(
      'a',
      { href: 'https://github.com/kirito-Lan/DeepLearningInStock/tree/front-end' },
      '关于',
    ),
    title: '关于',
  },
])
</script>

<style scoped>
.title-bar {
  display: flex;
  align-items: center;
}

.title {
  color: black;
  font-size: 18px;
  margin-left: 16px;
}

.logo {
  margin-left: 20px;
  height: 48px;
}
</style>
